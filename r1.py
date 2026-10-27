"""R1 tab: prompt A -> a text model writes several image prompts -> you pick one -> prompt B + that prompt = prompt C -> image.

  GET  /api/r1/options            text models, sizes, default Prompt B
  POST /api/r1/create             photos + name -> an R1 run, returns {folder}
  GET  /api/r1/{folder}           the whole run (photos, every prompt run with its tiles, every image run)
  POST /api/r1/{folder}/prompts   Prompt A + text model + effort + how many -> several prompts per photo (shown as tiles)
  POST /api/r1/{folder}/pick      choose which tile (prompt) of a photo is used for the image
  POST /api/r1/{folder}/image     Prompt B + the chosen tile -> Prompt C -> gpt-image-2.5-sunburst, low

Prompt C = Prompt B, a blank line, then the chosen prompt. Each photo gets its own tiles (the text model looks at that photo).
Everything is kept: every prompt run (with its cost) and every image (with the prompt that was chosen and its cost).
Stored in outputs/r1_<n>_<name>/r1.json and listed in History.
"""
import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import batch
import main as core
import scene_library as lib
from costing import estimate_inr, log_text_cost, text_rate

router = APIRouter()

HERE = Path(__file__).parent
DEFAULT_B = (HERE / "r1_prompt_b.txt").read_text(encoding="utf-8")
IMAGE_MODEL, QUALITY = batch.BATCH_MODEL, batch.BATCH_QUALITY
_FOLDER = re.compile(r"^r1_\d+_[A-Za-z0-9_.+\-]+$")
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=int(__import__("os").getenv("WEB_BATCH_WORKERS", "3")))
JOBS: dict = {}
MAX_PHOTOS = 30
EST_IN = 2500                                                           # rough tokens for the price preview only
EST_OUT = {"none": 500, "minimal": 600, "low": 900, "medium": 1800, "high": 3500, "xhigh": 7000, "max": 12000}

R1_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after):\n'
    '{"prompts": ["first prompt", "second prompt", "..."]}\n'
    'Each item of the list is one complete prompt, written as one plain string. The prompts must differ from one another.'
)
USER_TEXT = "Write the prompts for the product shown in the attached photo, as instructed."


def _now() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _models():
    from webapp import TEXT_MODELS, TEXT_PROVIDER, _text_ready
    return TEXT_MODELS, TEXT_PROVIDER, _text_ready


# ---------------------------------------------------------------- storage
def _file(folder: str) -> Path:
    return core.OUTPUT_DIR / folder / "r1.json"


def _save(job: dict):
    p = _file(job["folder"])
    tmp = p.with_suffix(".tmp")
    with LOCK:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def _get(folder: str) -> dict:
    if not _FOLDER.match(folder) or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "R1 run not found.")
    with LOCK:
        if folder in JOBS:
            return JOBS[folder]
        try:
            job = json.loads(_file(folder).read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(404, "R1 run not found.")
        for it in job["items"]:
            if it.get("running"):                                       # the server restarted while it was working
                it["running"], it["error"] = "", "Stopped: the server restarted before this finished."
        JOBS[folder] = job
        return job


def _new_folder(name: str) -> Path:
    with LOCK:
        nums = [int(p.name.split("_")[1]) for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and _FOLDER.match(p.name)]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:24] or "r1"
        d = core.OUTPUT_DIR / f"r1_{max(nums, default=0) + 1}_{slug}"
        d.mkdir()
    return d


def _view(job: dict) -> dict:
    with LOCK:
        out = copy.deepcopy(job)
    f = job["folder"]
    for it in out["items"]:
        it["input_url"] = f"/outputs/{f}/{it['input']}"
        for r in it["images"]:
            r["url"] = f"/outputs/{f}/{r['file']}"
    return out


def _image_of(folder: str, it: dict):
    name = it["input"]
    ctype = "image/png" if name.lower().endswith(".png") else "image/webp" if name.lower().endswith(".webp") else "image/jpeg"
    return (name, (core.OUTPUT_DIR / folder / name).read_bytes(), ctype)


def _check_model(text_model: str, effort: str):
    tm, prov, ready = _models()
    if text_model not in tm:
        raise HTTPException(400, "Pick a text model.")
    if not ready(text_model):
        raise HTTPException(400, f"{text_model} needs {'GEMINI_API_KEY' if prov[text_model] == 'google' else 'MODEL_API_KEY'} in the server's .env file.")
    if effort not in tm[text_model]["efforts"]:
        raise HTTPException(400, f"{text_model} does not support effort '{effort}'. Supported: {', '.join(tm[text_model]['efforts'])}.")


def _targets(job: dict, indexes: str) -> list:
    n = len(job["items"])
    if indexes.strip():
        try:
            todo = sorted({int(x) for x in indexes.split(",") if x.strip() != ""})
        except ValueError:
            raise HTTPException(400, "Bad photo list.")
        if any(i < 0 or i >= n for i in todo):
            raise HTTPException(400, "Unknown photo.")
    else:
        todo = list(range(n))
    with LOCK:
        busy = [i + 1 for i in todo if job["items"][i].get("running")]
    if busy:
        raise HTTPException(409, f"Photo {', '.join(map(str, busy))} is still working. Wait for it to finish.")
    return todo


def _begin(job: dict, todo: list, kind: str):
    with LOCK:
        for i in todo:
            job["items"][i].update(running=kind, error="")
        _save(job)


def _end(job: dict, i: int, error: str = ""):
    with LOCK:
        job["items"][i].update(running="", error=error)
        _save(job)


# ---------------------------------------------------------------- step 1: prompts
def _tiles_from(text: str) -> list:
    d = lib.parse_json(text)
    raw = d.get("prompts")
    if not isinstance(raw, list):
        raise ValueError('the JSON has no "prompts" list')
    out = []
    for x in raw:
        if isinstance(x, dict):
            x = x.get("prompt") or x.get("text") or next((v for v in x.values() if isinstance(v, str)), "")
        x = str(x or "").strip()
        if x:
            out.append(x)
    if not out:
        raise ValueError("the list of prompts is empty")
    return out


def _do_prompts(job: dict, i: int, cfg: dict):
    it, folder = job["items"][i], job["folder"]
    err, run_cost, tiles, last = "", 0.0, [], ""
    try:
        provider = _models()[1][cfg["text_model"]]
        instr = (cfg["prompt_a"].strip() + f"\n\nWrite exactly {cfg['count']} different prompts.\n\n" + R1_FORMAT)
        for attempt in (1, 2):
            extra = "" if attempt == 1 else (f"\n\nYour previous reply could not be used ({last}). "
                                             "Reply again with ONLY the JSON object described above.")
            text, usage = core.generate_text(provider, cfg["text_model"], cfg["effort"], instr, USER_TEXT + extra,
                                             _image_of(folder, it))
            cost, _, _ = log_text_cost(core.COST_LOG, cfg["text_model"], usage,
                                       note=f"r1 prompts, effort {cfg['effort']} | {folder}/{it['input']}")
            run_cost += cost
            with LOCK:
                job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            try:
                tiles = _tiles_from(text)
                break
            except Exception as e:
                last = str(e)[:120]
        else:
            raise RuntimeError(f"The text model did not return a usable list of prompts ({last}).")
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
    with LOCK:
        run = {"n": len(it["runs"]) + 1, "created": _now(), "text_model": cfg["text_model"], "effort": cfg["effort"],
               "prompt_a": cfg["prompt_a"], "count": cfg["count"], "tiles": tiles, "cost_inr": round(run_cost, 4)}
        if err:
            run["error"] = err
        if tiles or run_cost:                                           # a failed call that cost money is still shown
            it["runs"].append(run)
        job["prompt_a"] = cfg["prompt_a"]
    _end(job, i, err)


def _run_prompts(job: dict, todo: list, cfg: dict):
    try:
        list(POOL.map(lambda i: _do_prompts(job, i, cfg), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


# ---------------------------------------------------------------- step 2: image
def _do_image(job: dict, i: int, prompt_b: str, size: str):
    it, folder = job["items"][i], job["folder"]
    err = ""
    try:
        sel = it.get("sel") or {}
        run = next((r for r in it["runs"] if r["n"] == sel.get("run")), None)
        if not run or not (0 <= sel.get("tile", -1) < len(run["tiles"])):
            raise RuntimeError("Choose one of the prompts (tiles) for this photo first.")
        tile = run["tiles"][sel["tile"]]
        prompt_c = (prompt_b.strip() + "\n\n" + tile).strip()
        png, usage = core._generate(_image_of(folder, it), prompt_c, size, IMAGE_MODEL, QUALITY)
        n = len(it["images"]) + 1
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}_{n}.png"
        cost = core._store(core.OUTPUT_DIR / folder, fname, png, usage, size, IMAGE_MODEL, QUALITY, f"r1 image | {folder}")
        with LOCK:
            it["images"].append({"n": n, "created": _now(), "run_n": run["n"], "tile_idx": sel["tile"], "text_model": run["text_model"],
                                 "effort": run["effort"], "prompt_a_output": tile, "prompt_b": prompt_b, "prompt_c": prompt_c,
                                 "size": size, "file": fname, "cost_inr": round(cost, 4)})
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            job["size"], job["prompt_b"] = size, prompt_b
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
    _end(job, i, err)


def _run_images(job: dict, todo: list, prompt_b: str, size: str):
    try:
        list(POOL.map(lambda i: _do_image(job, i, prompt_b, size), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


# ---------------------------------------------------------------- API
@router.get("/api/r1/options")
def r1_options():
    from webapp import EFFORTS, SIZES
    tm, prov, ready = _models()
    return {
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"], "available": ready(k),
                         "needs": ("GEMINI_API_KEY" if prov[k] == "google" else "MODEL_API_KEY" if prov[k] == "meta" else "") + " in the server's .env",
                         "rate_in": text_rate(k)[0], "rate_out": text_rate(k)[1]} for k, v in tm.items()],
        "efforts": EFFORTS, "est_in": EST_IN, "est_out": EST_OUT, "usd_to_inr": core.USD_TO_INR,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "prompt_b": DEFAULT_B, "max_photos": MAX_PHOTOS,
        "image_model": IMAGE_MODEL, "quality": QUALITY, "est_image_inr_1024": round(estimate_inr(QUALITY, IMAGE_MODEL), 4),
    }


@router.post("/api/r1/create")
async def r1_create(images: list[UploadFile] = File(...), name: str = Form("")):
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > MAX_PHOTOS:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {MAX_PHOTOS} per run.")
    read = [await core._read_image(up) for up in images]
    d = _new_folder(name or "r1")
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "runs": [], "sel": None, "images": [], "running": "", "error": ""})
    job = {"folder": d.name, "flow": "r1", "name": name.strip() or "R1", "created": _now(), "size": "1088x1360",
           "prompt_a": "", "prompt_b": "", "total_inr": 0, "items": items}
    with LOCK:
        JOBS[d.name] = job
    _save(job)
    return {"folder": d.name, "photos": len(items)}


@router.get("/api/r1/{folder}")
def r1_get(folder: str):
    return _view(_get(folder))


@router.post("/api/r1/{folder}/prompts")
def r1_prompts(folder: str, prompt_a: str = Form(...), text_model: str = Form(...), effort: str = Form(...),
               count: int = Form(5), indexes: str = Form("")):
    job = _get(folder)
    prompt_a = prompt_a.strip()
    if not prompt_a:
        raise HTTPException(400, "Write Prompt A first.")
    if len(prompt_a) > 20000:
        raise HTTPException(400, "Prompt A is too long (limit 20,000 characters).")
    if count < 1:
        raise HTTPException(400, "Ask for at least 1 prompt.")
    _check_model(text_model, effort)
    todo = _targets(job, indexes)
    _begin(job, todo, "prompts")
    cfg = {"prompt_a": prompt_a, "text_model": text_model, "effort": effort, "count": count}
    threading.Thread(target=_run_prompts, args=(job, todo, cfg), daemon=True).start()
    return {"started": len(todo)}


class Pick(BaseModel):
    index: int
    run: int
    tile: int


@router.post("/api/r1/{folder}/pick")
def r1_pick(folder: str, body: Pick):
    job = _get(folder)
    if not 0 <= body.index < len(job["items"]):
        raise HTTPException(400, "Unknown photo.")
    it = job["items"][body.index]
    run = next((r for r in it["runs"] if r["n"] == body.run), None)
    if not run or not 0 <= body.tile < len(run["tiles"]):
        raise HTTPException(400, "Unknown prompt.")
    with LOCK:
        it["sel"] = {"run": body.run, "tile": body.tile}
        _save(job)
    return {"ok": True}


@router.post("/api/r1/{folder}/image")
def r1_image(folder: str, prompt_b: str = Form(""), size: str = Form("1088x1360"), indexes: str = Form("")):
    job = _get(folder)
    if len(prompt_b) > 20000:
        raise HTTPException(400, "Prompt B is too long (limit 20,000 characters).")
    core._check_size(size)
    todo = _targets(job, indexes)
    todo = [i for i in todo if job["items"][i].get("sel")]
    if not todo:
        raise HTTPException(400, "Choose a prompt (tile) for at least one photo first.")
    _begin(job, todo, "image")
    threading.Thread(target=_run_images, args=(job, todo, prompt_b.strip(), size), daemon=True).start()
    return {"started": len(todo)}


# ---------------------------------------------------------------- History
def r1_list() -> list:
    rows = []
    for p in core.OUTPUT_DIR.iterdir():
        if not (p.is_dir() and _FOLDER.match(p.name)):
            continue
        try:
            job = _get(p.name)
        except HTTPException:
            continue
        items = job["items"]
        imgs = [r for it in items for r in it["images"]]
        try:
            ts = datetime.strptime(job["created"], "%d %b %Y, %H:%M").timestamp()
        except Exception:
            ts = p.stat().st_mtime
        running = any(it.get("running") for it in items)
        thumb = next((f"/outputs/{p.name}/{r['file']}" for it in items for r in it["images"][-1:]), "") \
            or (f"/outputs/{p.name}/{items[0]['input']}" if items else "")
        rows.append({
            "type": "r1", "flow": "r1", "folder": p.name, "created": job["created"], "ts": ts,
            "title": job.get("name") or "R1", "state": "running" if running else "done", "thumb": thumb,
            "photos": len(items), "prompt_runs": sum(len(it["runs"]) for it in items), "images_done": len(imgs),
            "images_total": len(imgs), "failed": sum(1 for it in items if it.get("error")),
            "cost_inr": round(job.get("total_inr", 0), 4),
            "models": list(dict.fromkeys([f"{r['text_model']} ({r['effort']})" for it in items for r in it["runs"]] + [IMAGE_MODEL])),
            "categories": [],
            "selected": [{"photo": it["orig_name"], "text": it["images"][-1]["prompt_a_output"]} for it in items if it["images"]],
        })
    return rows
