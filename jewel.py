"""Jewellery tab: one of 15 scene prompts + a photo + a text model -> ONE scene -> (scene + your image prompt + the photo) -> flare / sunburst image.

  GET  /api/jw/options             15 prompts, text models, image models (flare low, sunburst low), sizes, default image prompt
  POST /api/jw/create              photos + the chosen prompt -> a run, returns {folder}
  GET  /api/jw/{folder}            the whole run
  POST /api/jw/{folder}/scene      step 1: write a scene (each call writes a NEW, different scene)
  POST /api/jw/{folder}/use        choose which written scene of a photo goes to the image step
  POST /api/jw/{folder}/image      step 2: the chosen scene + image prompt + photo -> image

WHY THE SCENES DIFFER FROM RUN TO RUN WITHOUT A SCENE LIBRARY
Before every scene call the app "deals" a creative brief: a setting (from 8 per prompt), a colour mood, a light, a camera, a presentation and
a prop style. For the same photo + same prompt it always picks the choices used the LEAST so far (ties at random), by looking at the earlier
runs of that photo in History. So the setting changes every run, and the other choices rotate too. The model also gets the opening lines of
the scenes already written for that photo and prompt, with "never repeat". Nothing is stored apart from the normal run files.
Stored in outputs/jw_<n>_<name>/jw.json and listed in History.
"""
import copy
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import batch
import imgchoices
import main as core
from costing import log_text_cost, text_rate

router = APIRouter()
HERE = Path(__file__).parent
_DATA = json.loads((HERE / "jewel_prompts.json").read_text(encoding="utf-8"))
PROMPTS = {p["id"]: p for p in _DATA["prompts"]}
SHARED = _DATA["shared"]
DEFAULT_IMAGE_PROMPT = (HERE / "jewel_image_prompt.txt").read_text(encoding="utf-8")
IMAGE_CHOICES = ("flare-low", "sunburst-low")
_FOLDER = re.compile(r"^jw_\d+_[A-Za-z0-9_.+\-]+$")
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=int(__import__("os").getenv("WEB_BATCH_WORKERS", "3")))
JOBS: dict = {}
MAX_PHOTOS = 20
EST_IN = 3200
EST_OUT = {"none": 400, "minimal": 500, "low": 700, "medium": 1200, "high": 2200, "xhigh": 4000, "max": 7000}
USER_TEXT = "Write the scene for the jewellery in the attached photo, as instructed."
AVOID_MAX = 8


def _now() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _models():
    from webapp import TEXT_MODELS, TEXT_PROVIDER, _text_ready
    return TEXT_MODELS, TEXT_PROVIDER, _text_ready


# ---------------------------------------------------------------- storage
def _file(folder: str) -> Path:
    return core.OUTPUT_DIR / folder / "jw.json"


def _save(job: dict):
    p = _file(job["folder"])
    tmp = p.with_suffix(".tmp")
    with LOCK:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def _get(folder: str) -> dict:
    if not _FOLDER.match(folder) or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "Jewellery run not found.")
    with LOCK:
        if folder in JOBS:
            return JOBS[folder]
        try:
            job = json.loads(_file(folder).read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(404, "Jewellery run not found.")
        for it in job["items"]:
            if it.get("running"):
                it["running"], it["error"] = "", "Stopped: the server restarted before this finished."
        JOBS[folder] = job
        return job


def _new_folder(name: str) -> Path:
    with LOCK:
        nums = [int(p.name.split("_")[1]) for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and _FOLDER.match(p.name)]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:24] or "jewellery"
        d = core.OUTPUT_DIR / f"jw_{max(nums, default=0) + 1}_{slug}"
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


# ---------------------------------------------------------------- the creative brief (what keeps every run different)
def _past_runs(image_key: str, prompt_id: int) -> list:
    """Every earlier scene run for this exact photo with this prompt, from all runs on disk (and the live ones)."""
    out = []
    for p in core.OUTPUT_DIR.iterdir():
        if not (p.is_dir() and _FOLDER.match(p.name)):
            continue
        try:
            job = JOBS.get(p.name) or json.loads(_file(p.name).read_text(encoding="utf-8"))
        except Exception:
            continue
        for it in job.get("items", []):
            if it.get("image_key") == image_key:
                out += [r for r in it.get("runs", []) if r.get("prompt_id") == prompt_id and r.get("scene")]
    return out


def _least_used(pool: list, used: list) -> str:
    counts = {x: used.count(x) for x in pool}
    low = min(counts.values())
    return random.choice([x for x in pool if counts[x] == low])


def deal_brief(image_key: str, prompt_id: int) -> tuple[dict, list]:
    """Pick the setting, colours, light, camera, presentation and props this run must use, plus the earlier scenes to avoid."""
    p = PROMPTS[prompt_id]
    past = _past_runs(image_key, prompt_id)
    briefs = [r.get("brief", {}) for r in past]
    pick = lambda key, pool: _least_used(pool, [b.get(key) for b in briefs])
    brief = {"setting": pick("setting", p["settings"]), "colours": pick("colours", SHARED["palettes"]),
             "light": pick("light", SHARED["lights"]), "camera": pick("camera", p.get("cameras") or SHARED["cameras"]),
             "presentation": pick("presentation", SHARED["presentations"]), "props": pick("props", SHARED["props"])}
    avoid = [re.sub(r"\s+", " ", r["scene"]).strip()[:170] for r in past][-AVOID_MAX:]
    return brief, avoid


def brief_text(brief: dict, avoid: list) -> str:
    lines = ["CREATIVE BRIEF FOR THIS RUN (chosen by the app so that every run is different; follow it)",
             f"- Setting and spot: {brief['setting']}",
             f"- Colour mood: {brief['colours']}",
             f"- Light: {brief['light']}",
             f"- Camera: {brief['camera']}",
             f"- Presentation of the jewellery: {brief['presentation']}",
             f"- Props: {brief['props']}",
             "",
             "SCENES ALREADY WRITTEN FOR THIS PHOTO WITH THIS PROMPT (the opening of each; never repeat any of them, not even with small changes):"]
    lines += [f"{k}. {a}" for k, a in enumerate(avoid, start=1)] or ["none yet"]
    return "\n".join(lines)


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t).strip()
    return t


# ---------------------------------------------------------------- step 1: the scene
def _do_scene(job: dict, i: int, cfg: dict):
    it, folder = job["items"][i], job["folder"]
    err, cost_total, scene, instr, brief, avoid = "", 0.0, "", "", {}, []
    try:
        provider = _models()[1][cfg["text_model"]]
        brief, avoid = deal_brief(it["image_key"], cfg["prompt_id"])
        instr = cfg["prompt_text"].strip() + "\n\n" + brief_text(brief, avoid)
        for attempt in (1, 2):
            text, usage = core.generate_text(provider, cfg["text_model"], cfg["effort"], instr, USER_TEXT, _image_of(folder, it))
            cost, _, _ = log_text_cost(core.COST_LOG, cfg["text_model"], usage, note=f"jewellery scene, effort {cfg['effort']} | {folder}/{it['input']}")
            cost_total += cost
            with LOCK:
                job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            scene = _clean(text)
            if scene:
                break
        if not scene:
            raise RuntimeError("The text model returned an empty scene.")
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
    with LOCK:
        run = {"n": len(it["runs"]) + 1, "created": _now(), "prompt_id": cfg["prompt_id"], "prompt_title": PROMPTS[cfg["prompt_id"]]["title"],
               "prompt_text": cfg["prompt_text"], "text_model": cfg["text_model"], "effort": cfg["effort"], "brief": brief, "avoid": avoid,
               "instructions_sent": instr, "scene": scene, "cost_inr": round(cost_total, 4)}
        if err:
            run["error"] = err
        if scene or cost_total:
            it["runs"].append(run)
            if scene:
                it["cur"] = run["n"]
    _end(job, i, err)


def _run_scenes(job: dict, todo: list, cfg: dict):
    try:
        list(POOL.map(lambda i: _do_scene(job, i, cfg), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


# ---------------------------------------------------------------- step 2: the image
def _do_image(job: dict, i: int, image_prompt: str, size: str, choice: str):
    it, folder = job["items"][i], job["folder"]
    err = ""
    try:
        model, quality = imgchoices.pick(choice)
        run = next((r for r in it["runs"] if r["n"] == it.get("cur")), None)
        if not run or not run.get("scene"):
            raise RuntimeError("Write a scene for this photo first.")
        prompt_c = (image_prompt.strip() + "\n" + run["scene"]).strip()
        with LOCK:
            n = max(it.get("img_seq", 0), len(it["images"])) + 1
            it["img_seq"] = n
        png, usage = core._generate(_image_of(folder, it), prompt_c, size, model, quality)
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}_{n}.png"
        cost = core._store(core.OUTPUT_DIR / folder, fname, png, usage, size, model, quality, f"jewellery image | {folder}")
        with LOCK:
            it["images"].append({"n": n, "created": _now(), "run_n": run["n"], "prompt_id": run["prompt_id"], "prompt_title": run["prompt_title"],
                                 "text_model": run["text_model"], "effort": run["effort"], "scene_cost_inr": run["cost_inr"], "scene": run["scene"],
                                 "scene_prompt": run["instructions_sent"], "image_prompt": image_prompt, "prompt_c": prompt_c, "size": size,
                                 "image_model": model, "quality": quality, "file": fname, "cost_inr": round(cost, 4)})
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            job["size"], job["image_prompt"] = size, image_prompt
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
    _end(job, i, err)


def _run_images(job: dict, todo: list, image_prompt: str, size: str, choice: str):
    try:
        list(POOL.map(lambda i: _do_image(job, i, image_prompt, size, choice), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


# ---------------------------------------------------------------- API
def _choices() -> list:
    return [c for c in imgchoices.for_page() if c["id"] in IMAGE_CHOICES]


@router.get("/api/jw/options")
def jw_options():
    from webapp import SIZES
    tm, prov, ready = _models()
    return {
        "prompts": [{"id": p["id"], "title": p["title"], "summary": p["summary"], "prompt": p["prompt"]} for p in PROMPTS.values()],
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"], "available": ready(k),
                         "needs": ("GEMINI_API_KEY" if prov[k] == "google" else "MODEL_API_KEY" if prov[k] == "meta" else "") + " in the server's .env",
                         "rate_in": text_rate(k)[0], "rate_out": text_rate(k)[1]} for k, v in tm.items()],
        "est_in": EST_IN, "est_out": EST_OUT, "usd_to_inr": core.USD_TO_INR,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "image_choices": _choices(), "image_prompt": DEFAULT_IMAGE_PROMPT, "max_photos": MAX_PHOTOS,
    }


@router.post("/api/jw/create")
async def jw_create(images: list[UploadFile] = File(...), name: str = Form("")):
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > MAX_PHOTOS:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {MAX_PHOTOS} per run.")
    read = [await core._read_image(up) for up in images]
    d = _new_folder(name or "jewellery")
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "image_key": hashlib.md5(img[1]).hexdigest(), "runs": [], "cur": 0,
                      "images": [], "running": "", "error": ""})
    job = {"folder": d.name, "flow": "jw", "name": name.strip() or "Jewellery", "created": _now(), "size": "1088x1360",
           "prompt_id": 0, "image_prompt": "", "total_inr": 0, "items": items}
    with LOCK:
        JOBS[d.name] = job
    _save(job)
    return {"folder": d.name, "photos": len(items)}


@router.get("/api/jw/{folder}")
def jw_get(folder: str):
    return _view(_get(folder))


@router.post("/api/jw/{folder}/scene")
def jw_scene(folder: str, prompt_id: int = Form(...), prompt_text: str = Form(...), text_model: str = Form(...), effort: str = Form(...),
             indexes: str = Form("")):
    job = _get(folder)
    if prompt_id not in PROMPTS:
        raise HTTPException(400, "Choose one of the 15 prompts.")
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise HTTPException(400, "The prompt is empty.")
    if len(prompt_text) > 30000:
        raise HTTPException(400, "The prompt is too long (limit 30,000 characters).")
    _check_model(text_model, effort)
    todo = _targets(job, indexes)
    _begin(job, todo, "scene")
    with LOCK:
        job["prompt_id"] = prompt_id
    cfg = {"prompt_id": prompt_id, "prompt_text": prompt_text, "text_model": text_model, "effort": effort}
    threading.Thread(target=_run_scenes, args=(job, todo, cfg), daemon=True).start()
    return {"started": len(todo)}


class Use(BaseModel):
    index: int
    run: int


@router.post("/api/jw/{folder}/use")
def jw_use(folder: str, body: Use):
    job = _get(folder)
    if not 0 <= body.index < len(job["items"]):
        raise HTTPException(400, "Unknown photo.")
    it = job["items"][body.index]
    if not any(r["n"] == body.run and r.get("scene") for r in it["runs"]):
        raise HTTPException(400, "Unknown scene.")
    with LOCK:
        it["cur"] = body.run
        _save(job)
    return {"ok": True}


@router.post("/api/jw/{folder}/image")
def jw_image(folder: str, image_prompt: str = Form(""), size: str = Form("1088x1360"), image_choice: str = Form(...), indexes: str = Form("")):
    job = _get(folder)
    if image_choice not in IMAGE_CHOICES:
        raise HTTPException(400, "Choose the image model.")
    if len(image_prompt) > 30000:
        raise HTTPException(400, "The image prompt is too long (limit 30,000 characters).")
    core._check_size(size)
    todo = [i for i in _targets(job, indexes) if job["items"][i].get("cur")]
    if not todo:
        raise HTTPException(400, "Write a scene for at least one photo first.")
    _begin(job, todo, "image")
    threading.Thread(target=_run_images, args=(job, todo, image_prompt.strip(), size, image_choice), daemon=True).start()
    return {"started": len(todo)}


# ---------------------------------------------------------------- History
def list_rows() -> list:
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
        thumb = next((f"/outputs/{p.name}/{r['file']}" for it in items for r in it["images"][-1:]), "") \
            or (f"/outputs/{p.name}/{items[0]['input']}" if items else "")
        rows.append({
            "type": "jw", "flow": "jw", "folder": p.name, "created": job["created"], "ts": ts, "title": job.get("name") or "Jewellery",
            "state": "running" if any(it.get("running") for it in items) else "done", "thumb": thumb,
            "photos": len(items), "scene_runs": sum(len(it["runs"]) for it in items), "images_done": len(imgs), "images_total": len(imgs),
            "failed": sum(1 for it in items if it.get("error")), "cost_inr": round(job.get("total_inr", 0), 4),
            "models": list(dict.fromkeys([f"{r['text_model']} ({r['effort']})" for it in items for r in it["runs"]] +
                                         [f"{r['image_model']} {r['quality']}" for r in imgs])),
            "categories": [], "selected": [{"photo": it["orig_name"], "text": r["scene"]} for it in items for r in it["runs"][-1:] if r.get("scene")][:8],
        })
    return rows
