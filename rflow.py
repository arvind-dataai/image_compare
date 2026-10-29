"""R2 and R3 tabs: identify -> scenes -> image, each step run on its own, each step can be run again.

  GET  /api/rf/{kind}/options          text models, sizes, default prompts
  GET  /api/rf/{kind}/library          this tab's own scene library (category > subcategory > scenes)
  POST /api/rf/{kind}/create           photos + name -> a run, returns {folder}
  GET  /api/rf/{kind}/{folder}         the whole run
  POST /api/rf/{kind}/{folder}/identify   step 1: prompt + text model -> category, subcategory (existing or new)
  POST /api/rf/{kind}/{folder}/useid      choose which step-1 result of a photo is the current one
  POST /api/rf/{kind}/{folder}/scenes     step 2: prompt + text model -> several scenes (the existing scenes of that subcategory are analysed)
  POST /api/rf/{kind}/{folder}/pick       choose the scene (tile) that goes to step 3
  POST /api/rf/{kind}/{folder}/image      step 3: your prompt + the chosen scene -> gpt-image-2.5-sunburst, low

kind = "r2", "r3" or "r4". Each has its OWN scene library file (r2_library.json / r3_library.json / r4_library.json).
R4 = R2 with no prompt in step 3 and a choice of image model (flare or sunburst) and quality (low or medium).

  R2: the scenes the AI writes are shown as tiles and nothing is saved until you pick one: the PICKED scene is saved in the library.
  R3: every scene the AI writes is saved in the library at once (status "unused"). Step 2 shows the unused scenes of the photo's
      category/subcategory from the library; you pick one; it becomes "used" when its image is made. Writing new scenes is optional.

Step 3 prompt = your prompt, a blank line, then the chosen scene. The photo is sent together with it.
Stored in outputs/<kind>_<n>_<name>/job.json and listed in History.
"""
import copy
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import batch
import main as core
import scene_library as sl
from costing import estimate_inr, log_text_cost, text_rate

router = APIRouter()

HERE = Path(__file__).parent
DEFAULT_ID = (HERE / "scene_classify_prompt.txt").read_text(encoding="utf-8")
DEFAULT_SCENE = (HERE / "r_scene_prompt.txt").read_text(encoding="utf-8")
DEFAULT_B = (HERE / "r1_prompt_b.txt").read_text(encoding="utf-8")
IMAGE_MODEL, QUALITY = batch.BATCH_MODEL, batch.BATCH_QUALITY
KINDS = ("r2", "r3", "r4")
R2LIKE = ("r2", "r4")                 # R4 behaves exactly like R2 (own library), except for step 3
# R4 step 3: no prompt box; the image model and quality are chosen in the page
R4_CHOICES = {"flare-low": ("gpt-image-2.5-flare", "low"), "flare-medium": ("gpt-image-2.5-flare", "medium"),
              "sunburst-low": ("gpt-image-2.5-sunburst", "low"), "sunburst-medium": ("gpt-image-2.5-sunburst", "medium")}
_FOLDER = re.compile(r"^(r2|r3|r4)_\d+_[A-Za-z0-9_.+\-]+$")
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=int(__import__("os").getenv("WEB_BATCH_WORKERS", "3")))
JOBS: dict = {}
MAX_PHOTOS = 30
CONTEXT_LIMIT = 80                                                      # most recent existing scenes sent to the text model
EST = {"id": 2500, "scene": 4000}                                       # rough input tokens for the price preview only
EST_OUT = {"none": 500, "minimal": 600, "low": 900, "medium": 1800, "high": 3500, "xhigh": 7000, "max": 12000}

CLASSIFY_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after):\n'
    '{"category": "...", "subcategory": "...", "product_description": "two or three plain sentences about the product as described above", '
    '"reason": "one or two sentences: why it belongs there and which existing option you considered closest"}'
)
SCENES_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after):\n'
    '{"scenes": ["first scene", "second scene", "..."]}\n'
    'Each item of the list is ONE complete scene written as one string, laid out exactly as the instructions above describe '
    '(use \\n between its lines). Write as many scenes as the instructions above ask for; if they give no number, write 8.'
)


def _now() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _models():
    from webapp import TEXT_MODELS, TEXT_PROVIDER, _text_ready
    return TEXT_MODELS, TEXT_PROVIDER, _text_ready


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise HTTPException(404, "Unknown tab.")
    return kind


# ---------------------------------------------------------------- the scene library (one file per tab)
class Lib:
    def __init__(self, kind: str):
        self.kind = kind
        self.file = Path(core.COST_LOG).parent / f"{kind}_library.json"
        self.lock = threading.RLock()

    def _load(self) -> dict:
        try:
            d = json.loads(self.file.read_text(encoding="utf-8"))
            d.setdefault("categories", {})
            return d
        except Exception:
            return {"categories": {}}

    def _save(self, d: dict):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def taxonomy(self) -> dict:
        with self.lock:
            d = self._load()
        return {c: list(v.get("subcategories", {})) for c, v in d["categories"].items()}

    def taxonomy_text(self) -> str:
        t = self.taxonomy()
        if not t:
            return "(the library is empty: no categories yet, so create the first category and subcategory)"
        return "\n".join(f"- {c}: " + (", ".join(s) if s else "(no subcategories yet)") for c, s in t.items())

    def resolve(self, category: str, subcategory: str) -> tuple:
        """(category, subcategory, new_category, new_subcategory); existing spellings win."""
        category, subcategory = sl.clean_name(category), sl.clean_name(subcategory)
        t = self.taxonomy()
        cat = next((c for c in t if sl.key_of(c) == sl.key_of(category)), None)
        if cat is None:
            return category, subcategory, True, True
        sub = next((s for s in t[cat] if sl.key_of(s) == sl.key_of(subcategory)), None)
        return cat, (sub or subcategory), False, sub is None

    def ensure(self, category: str, subcategory: str):
        with self.lock:
            d = self._load()
            d["categories"].setdefault(category, {"subcategories": {}})["subcategories"].setdefault(subcategory, {"scenes": []})
            self._save(d)

    def scenes(self, category: str, subcategory: str) -> list:
        with self.lock:
            d = self._load()
        return list(d["categories"].get(category, {}).get("subcategories", {}).get(subcategory, {}).get("scenes", []))

    def add(self, category: str, subcategory: str, entry: dict) -> str:
        entry = dict(entry, id=uuid.uuid4().hex[:8], created=datetime.now().isoformat(timespec="seconds"))
        with self.lock:
            d = self._load()
            sub = d["categories"].setdefault(category, {"subcategories": {}})["subcategories"].setdefault(subcategory, {"scenes": []})
            sub["scenes"].append(entry)
            self._save(d)
        return entry["id"]

    def get(self, scene_id: str) -> dict | None:
        with self.lock:
            d = self._load()
        for c in d["categories"].values():
            for s in c.get("subcategories", {}).values():
                for sc in s.get("scenes", []):
                    if sc.get("id") == scene_id:
                        return dict(sc)
        return None

    def update(self, scene_id: str, **fields) -> bool:
        with self.lock:
            d = self._load()
            for c in d["categories"].values():
                for s in c.get("subcategories", {}).values():
                    for sc in s.get("scenes", []):
                        if sc.get("id") == scene_id:
                            sc.update(fields)
                            self._save(d)
                            return True
        return False

    def overview(self) -> list:
        with self.lock:
            d = self._load()
        return [{"name": c, "subcategories": [{"name": s, "scenes": [
            {k: x.get(k, "") for k in ("id", "created", "status", "text", "photo", "folder", "file", "text_model", "effort")}
            for x in sv.get("scenes", [])]} for s, sv in cv.get("subcategories", {}).items()]}
            for c, cv in d["categories"].items()]


LIBS = {k: Lib(k) for k in KINDS}


# ---------------------------------------------------------------- storage
def _save(job: dict):
    p = core.OUTPUT_DIR / job["folder"] / "job.json"
    tmp = p.with_suffix(".tmp")
    with LOCK:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def _get(kind: str, folder: str) -> dict:
    if not _FOLDER.match(folder) or not folder.startswith(kind + "_") or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "Run not found.")
    with LOCK:
        if folder in JOBS:
            return JOBS[folder]
        try:
            job = json.loads((core.OUTPUT_DIR / folder / "job.json").read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(404, "Run not found.")
        for it in job["items"]:
            if "sel" in it:                                             # older runs kept one chosen scene: now it is a list
                old = it.pop("sel")
                it["sels"] = [old] if old else []
            it.setdefault("sels", [])
            if it.get("running"):
                it["running"], it["error"], it["pending"] = "", "Stopped: the server restarted before this finished.", 0
        JOBS[folder] = job
        return job


def _new_folder(kind: str, name: str) -> Path:
    with LOCK:
        nums = [int(p.name.split("_")[1]) for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith(kind + "_") and _FOLDER.match(p.name)]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:24] or kind
        d = core.OUTPUT_DIR / f"{kind}_{max(nums, default=0) + 1}_{slug}"
        d.mkdir()
    return d


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


def _text_call(job: dict, label: str, cfg: dict, instr: str, user_text: str, image, parse, note: str):
    """One text call (+ one retry if the reply is unusable). Returns (parsed, cost). Raises with .cost attached on failure."""
    provider = _models()[1][cfg["text_model"]]
    total, last = 0.0, ""
    for attempt in (1, 2):
        extra = "" if attempt == 1 else (f"\n\nYour previous reply could not be used ({last}). Reply again with ONLY the JSON object described above.")
        text, usage = core.generate_text(provider, cfg["text_model"], cfg["effort"], instr, user_text + extra, image)
        cost, _, _ = log_text_cost(core.COST_LOG, cfg["text_model"], usage, note=f"{job['kind']} {label}, effort {cfg['effort']} | {note}")
        total += cost
        with LOCK:
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
        try:
            return parse(text), total
        except Exception as e:
            last = str(e)[:120]
    err = RuntimeError(f"The text model did not return a usable reply for step {label} ({last}).")
    err.cost = total
    raise err


# ---------------------------------------------------------------- step 1: identify
def _parse_id(text: str) -> dict:
    d = sl.parse_json(text) if hasattr(sl, "parse_json") else json.loads(text)
    missing = [k for k in sl.CLASSIFY_KEYS if not str(d.get(k, "")).strip()]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return d


def _do_identify(job: dict, i: int, cfg: dict):
    kind, it, folder = job["kind"], job["items"][i], job["folder"]
    lib = LIBS[kind]
    run = {"n": len(it["id_runs"]) + 1, "created": _now(), "text_model": cfg["text_model"], "effort": cfg["effort"], "prompt": cfg["prompt"],
           "cost_inr": 0.0}
    err = ""
    try:
        instr = cfg["prompt"].strip() + "\n\nKNOWN TAXONOMY (categories and their subcategories already in the scene library):\n" + lib.taxonomy_text() + "\n\n" + CLASSIFY_FORMAT
        d, cost = _text_call(job, "1 identify", cfg, instr, "Identify the single product shown in the attached photo and file it, as instructed.",
                             _image_of(folder, it), _parse_id, f"{folder}/{it['input']}")
        run["cost_inr"] = round(cost, 4)
        with LOCK:                                                     # settle the names one photo at a time
            cat, sub, new_cat, new_sub = lib.resolve(str(d["category"]), str(d["subcategory"]))
            lib.ensure(cat, sub)
        run.update(category=cat, subcategory=sub, new_category=new_cat, new_subcategory=new_sub,
                   raw_category=str(d["category"]), raw_subcategory=str(d["subcategory"]),
                   description=str(d["product_description"]).strip(), reason=str(d.get("reason", "")).strip())
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
        run["cost_inr"] = round(getattr(e, "cost", 0.0), 4)
        run["error"] = err
    with LOCK:
        if run["cost_inr"] or not err:
            it["id_runs"].append(run)
        if not err:
            it["id_cur"] = run["n"]
            it["sels"] = []                                            # a new identification clears the chosen scenes (history is kept)
    _end(job, i, err)


def _run_pool(fn, job: dict, todo: list, args: tuple):
    try:
        list(POOL.map(lambda i: fn(job, i, *args), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


def _cur_id(it: dict) -> dict | None:
    return next((r for r in it["id_runs"] if r["n"] == it.get("id_cur") and not r.get("error")), None)


# ---------------------------------------------------------------- step 2: scenes
def _parse_scenes(text: str) -> list:
    d = sl.parse_json(text)
    raw = d.get("scenes") or d.get("prompts")
    if not isinstance(raw, list):
        raise ValueError('the JSON has no "scenes" list')
    out = []
    for x in raw:
        if isinstance(x, dict):                                        # a model that returns the fields as an object: join them
            x = "\n".join(f"{str(k).replace('_', ' ').upper()}: {v}" for k, v in x.items() if str(v).strip())
        elif isinstance(x, list):
            x = "\n".join(str(v) for v in x)
        x = str(x or "").strip()
        if x:
            out.append(x)
    if not out:
        raise ValueError("the list of scenes is empty")
    return out


def _context_scenes(kind: str, cat: str, sub: str) -> list:
    sc = LIBS[kind].scenes(cat, sub)
    if kind in R2LIKE:
        sc = [s for s in sc if s.get("status") in ("selected", "done")]
    else:
        sc = [s for s in sc if s.get("status") in ("unused", "used")]
    return sc[-CONTEXT_LIMIT:]


_SUBLOCKS: dict = {}


def _do_scenes(job: dict, i: int, cfg: dict):
    kind, it = job["kind"], job["items"][i]
    cur = _cur_id(it)
    if not cur:
        _end(job, i, "Run step 1 for this photo first.")
        return
    if kind == "r3":                      # R3 saves what it writes: photos of one subcategory go one after the other, so each sees the scenes just written
        with LOCK:
            lk = _SUBLOCKS.setdefault((cur["category"], cur["subcategory"]), threading.Lock())
        with lk:
            _scenes_body(job, i, cfg, cur)
    else:
        _scenes_body(job, i, cfg, cur)


def _scenes_body(job: dict, i: int, cfg: dict, cur: dict):
    kind, it, folder = job["kind"], job["items"][i], job["folder"]
    lib = LIBS[kind]
    cat, sub = cur["category"], cur["subcategory"]
    existing = _context_scenes(kind, cat, sub)
    run = {"n": len(it["scene_runs"]) + 1, "created": _now(), "text_model": cfg["text_model"], "effort": cfg["effort"], "prompt": cfg["prompt"],
           "id_run": cur["n"], "category": cat, "subcategory": sub, "existing_count": len(existing), "tiles": [], "lib_ids": [], "cost_inr": 0.0}
    err = ""
    try:
        listing = ("\n".join(f"{k}. " + s["text"].replace("\n", " | ") for k, s in enumerate(existing, start=1))
                   if existing else "None yet: there are no scenes for this subcategory, so this is the first set.")
        ctx = (f"CONTEXT\nPRODUCT\ncategory: {cat}\nsubcategory: {sub}\ndescription: {cur['description']}\n\n"
               f"EXISTING SCENES IN THE SCENE LIBRARY FOR THIS SUBCATEGORY ({len(existing)}):\n{listing}")
        instr = cfg["prompt"].strip() + "\n\n" + ctx + "\n\n" + SCENES_FORMAT
        tiles, cost = _text_call(job, "2 scenes", cfg, instr, "Write the scenes for the product in the attached photo, as instructed.",
                                 _image_of(folder, it), _parse_scenes, f"{folder}/{it['input']}")
        run["cost_inr"] = round(cost, 4)
        run["tiles"] = tiles
        if kind == "r3":                                               # R3 saves every scene it writes, unused until an image is made from it
            run["lib_ids"] = [lib.add(cat, sub, {"status": "unused", "text": t, "photo": it["orig_name"], "folder": folder,
                                                 "text_model": cfg["text_model"], "effort": cfg["effort"], "run_n": run["n"]}) for t in tiles]
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
        run["cost_inr"] = round(getattr(e, "cost", 0.0), 4)
        run["error"] = err
    with LOCK:
        if run["tiles"] or run["cost_inr"]:
            it["scene_runs"].append(run)
    _end(job, i, err)


# ---------------------------------------------------------------- step 3: image
def _scene_of(kind: str, it: dict, sel: dict):
    """(scene text, library id, scene run or None, cost of the run that wrote it for THIS photo) of one chosen scene."""
    if kind in R2LIKE:
        run = next((r for r in it["scene_runs"] if r["n"] == sel.get("run")), None)
        if not run or not 0 <= sel.get("tile", -1) < len(run["tiles"]):
            raise RuntimeError("A chosen scene no longer exists.")
        return run["tiles"][sel["tile"]], sel.get("lib_id"), run["n"], run["cost_inr"]
    s = LIBS[kind].get(sel.get("lib_id", ""))
    if not s:
        raise RuntimeError("A chosen scene no longer exists.")
    run = next((r for r in it["scene_runs"] if s["id"] in r.get("lib_ids", [])), None)
    return s["text"], s["id"], (run["n"] if run else None), (run["cost_inr"] if run else 0.0)


def _task_done(job: dict, i: int, err: str):
    """One image task of a photo finished; the photo is free again when its last task is done."""
    with LOCK:
        it = job["items"][i]
        if err:
            it.setdefault("errs", []).append(err)
        it["pending"] = it.get("pending", 1) - 1
        last = it["pending"] <= 0
        errs = " | ".join(it.pop("errs", [])) if last else ""
    if last:
        _end(job, i, errs)


def _do_image(job: dict, i: int, sel: dict, gen: int, sub: int, subs: int, prompt_b: str, size: str, model: str = "", quality: str = ""):
    kind, it, folder = job["kind"], job["items"][i], job["folder"]
    model, quality = model or IMAGE_MODEL, quality or QUALITY
    lib = LIBS[kind]
    err, label = "", "A chosen scene"
    try:
        cur = _cur_id(it)
        if not cur:
            raise RuntimeError("Run step 1 for this photo first.")
        text, lib_id, run_n, scene_cost = _scene_of(kind, it, sel)
        label = f"Scene {sub}"
        if kind == "r3":
            s = lib.get(lib_id)
            key = f"{folder}/{it['input']}"
            if s.get("status") == "used" and s.get("photo_key") != key:
                raise RuntimeError("That scene was already used for another photo. Choose another scene.")
        prompt_c = (prompt_b.strip() + "\n\n" + text).strip()
        with LOCK:
            n = max(it.get("img_seq", 0), len(it["images"])) + 1
            it["img_seq"] = n
        png, usage = core._generate(_image_of(folder, it), prompt_c, size, model, quality)
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}_{n}.png"
        cost = core._store(core.OUTPUT_DIR / folder, fname, png, usage, size, model, quality, f"{kind} image | {folder}")
        id_cost = cur["cost_inr"]
        with LOCK:
            it["images"].append({"n": n, "gen": gen, "sub": sub, "subs": subs, "created": _now(), "category": cur["category"],
                                 "subcategory": cur["subcategory"], "id_run": cur["n"], "id_model": cur["text_model"],
                                 "id_effort": cur["effort"], "id_cost_inr": id_cost, "scene_run": run_n, "scene_cost_inr": scene_cost,
                                 "image_model": model, "quality": quality, "lib_id": lib_id, "scene_text": text, "prompt_b": prompt_b, "prompt_c": prompt_c, "size": size,
                                 "file": fname, "cost_inr": round(cost, 4)})
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            job["size"], job["prompt_b"] = size, prompt_b
            job["image_model"], job["quality"] = model, quality
        if lib_id:
            lib.update(lib_id, status="used" if kind == "r3" else "done", file=fname, folder=folder, photo=it["orig_name"],
                       photo_key=f"{folder}/{it['input']}")
    except Exception as e:
        err = f"{label}: " + str(getattr(e, "detail", None) or e)
    _task_done(job, i, err)


def _run_images(job: dict, tasks: list, prompt_b: str, size: str, model: str = "", quality: str = ""):
    try:
        list(POOL.map(lambda t: _do_image(job, t[0], t[1], t[2], t[3], t[4], prompt_b, size, model, quality), tasks))
    except Exception as e:
        for i in {t[0] for t in tasks}:
            if job["items"][i].get("running"):
                _end(job, i, str(e))


# ---------------------------------------------------------------- view
def _view(job: dict) -> dict:
    kind, f = job["kind"], job["folder"]
    with LOCK:
        out = copy.deepcopy(job)
    lib = LIBS[kind]
    for it in out["items"]:
        it["input_url"] = f"/outputs/{f}/{it['input']}"
        for r in it["images"]:
            r["url"] = f"/outputs/{f}/{r['file']}"
        cur = _cur_id({"id_runs": it["id_runs"], "id_cur": it.get("id_cur")})
        it["current"] = ({k: cur[k] for k in ("n", "category", "subcategory", "new_category", "new_subcategory", "description")} if cur else None)
        if cur:
            sc = lib.scenes(cur["category"], cur["subcategory"])
            it["lib_total"] = len(sc)
            if kind == "r3":
                own = {lid: r["n"] for r in it["scene_runs"] for lid in r.get("lib_ids", [])}
                key = f"{f}/{it['input']}"
                it["lib_unused"] = [{"id": s["id"], "text": s["text"], "created": s.get("created", ""), "run_n": own.get(s["id"]),
                                     "photo": s.get("photo", "")} for s in sc if s.get("status") == "unused"]
                it["lib_mine"] = [s["id"] for s in sc if s.get("status") == "used" and s.get("photo_key") == key]
            else:
                it["lib_total"] = len([s for s in sc if s.get("status") in ("selected", "done")])
        if kind == "r3":
            it["sel_texts"] = {}
            for sel in it.get("sels", []):
                s = lib.get(sel.get("lib_id", ""))
                it["sel_texts"][sel.get("lib_id", "")] = s["text"] if s else ""
    return out


# ---------------------------------------------------------------- API
@router.get("/api/rf/{kind}/options")
def rf_options(kind: str):
    _kind(kind)
    from webapp import EFFORTS, SIZES
    tm, prov, ready = _models()
    return {
        "kind": kind,
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"], "available": ready(k),
                         "needs": ("GEMINI_API_KEY" if prov[k] == "google" else "MODEL_API_KEY" if prov[k] == "meta" else "") + " in the server's .env",
                         "rate_in": text_rate(k)[0], "rate_out": text_rate(k)[1]} for k, v in tm.items()],
        "efforts": EFFORTS, "est_in": EST, "est_out": EST_OUT, "usd_to_inr": core.USD_TO_INR,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "prompt_id": DEFAULT_ID, "prompt_scene": DEFAULT_SCENE, "prompt_b": DEFAULT_B, "max_photos": MAX_PHOTOS,
        "image_model": IMAGE_MODEL, "quality": QUALITY, "est_image_inr_1024": round(estimate_inr(QUALITY, IMAGE_MODEL), 4),
        "image_choices": [{"id": k, "model": m, "quality": q, "label": f"{m} · {q}", "est_inr_1024": round(estimate_inr(q, m), 4)} for k, (m, q) in R4_CHOICES.items()] if kind == "r4" else [],
    }


@router.get("/api/rf/{kind}/library")
def rf_library(kind: str):
    return {"categories": LIBS[_kind(kind)].overview()}


@router.post("/api/rf/{kind}/create")
async def rf_create(kind: str, images: list[UploadFile] = File(...), name: str = Form("")):
    _kind(kind)
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > MAX_PHOTOS:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {MAX_PHOTOS} per run.")
    read = [await core._read_image(up) for up in images]
    d = _new_folder(kind, name or kind)
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "id_runs": [], "id_cur": None, "scene_runs": [], "sels": [],
                      "images": [], "running": "", "error": ""})
    job = {"folder": d.name, "kind": kind, "name": name.strip() or kind.upper(), "created": _now(), "size": "1088x1360",
           "prompt_id": "", "prompt_scene": "", "prompt_b": "", "total_inr": 0, "items": items}
    with LOCK:
        JOBS[d.name] = job
    _save(job)
    return {"folder": d.name, "photos": len(items)}


@router.get("/api/rf/{kind}/{folder}")
def rf_get(kind: str, folder: str):
    return _view(_get(_kind(kind), folder))


def _prompt_ok(text: str, what: str) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, f"Write the {what} prompt first.")
    if len(text) > 30000:
        raise HTTPException(400, "The prompt is too long (limit 30,000 characters).")
    return text


@router.post("/api/rf/{kind}/{folder}/identify")
def rf_identify(kind: str, folder: str, prompt: str = Form(...), text_model: str = Form(...), effort: str = Form(...), indexes: str = Form("")):
    job = _get(_kind(kind), folder)
    prompt = _prompt_ok(prompt, "identification")
    _check_model(text_model, effort)
    todo = _targets(job, indexes)
    _begin(job, todo, "identify")
    job["prompt_id"] = prompt
    cfg = {"prompt": prompt, "text_model": text_model, "effort": effort}
    threading.Thread(target=_run_pool, args=(_do_identify, job, todo, (cfg,)), daemon=True).start()
    return {"started": len(todo)}


class UseId(BaseModel):
    index: int
    run: int


@router.post("/api/rf/{kind}/{folder}/useid")
def rf_useid(kind: str, folder: str, body: UseId):
    job = _get(_kind(kind), folder)
    if not 0 <= body.index < len(job["items"]):
        raise HTTPException(400, "Unknown photo.")
    it = job["items"][body.index]
    r = next((x for x in it["id_runs"] if x["n"] == body.run and not x.get("error")), None)
    if not r:
        raise HTTPException(400, "Unknown result.")
    with LOCK:
        if it.get("id_cur") != body.run:
            it["id_cur"], it["sels"] = body.run, []
        _save(job)
    return {"ok": True}


@router.post("/api/rf/{kind}/{folder}/scenes")
def rf_scenes(kind: str, folder: str, prompt: str = Form(...), text_model: str = Form(...), effort: str = Form(...), indexes: str = Form("")):
    job = _get(_kind(kind), folder)
    prompt = _prompt_ok(prompt, "scene")
    _check_model(text_model, effort)
    todo = _targets(job, indexes)
    missing = [i + 1 for i in todo if not _cur_id(job["items"][i])]
    if missing:
        raise HTTPException(400, f"Run step 1 first for photo {', '.join(map(str, missing))}.")
    _begin(job, todo, "scenes")
    job["prompt_scene"] = prompt
    cfg = {"prompt": prompt, "text_model": text_model, "effort": effort}
    threading.Thread(target=_run_pool, args=(_do_scenes, job, todo, (cfg,)), daemon=True).start()
    return {"started": len(todo)}


class Pick(BaseModel):
    index: int
    run: int | None = None
    tile: int | None = None
    lib_id: str | None = None
    clear: bool = False


def _drop_sel(job: dict, it: dict, sel: dict, lib: "Lib"):
    """R2: a scene that was picked but never used for an image no longer counts as saved when it is un-picked."""
    if sel.get("lib_id"):
        prev = lib.get(sel["lib_id"])
        if prev and prev.get("status") == "selected":
            lib.update(sel["lib_id"], status="replaced")


@router.post("/api/rf/{kind}/{folder}/pick")
def rf_pick(kind: str, folder: str, body: Pick):
    """Click a tile: choose it, click again: un-choose it. Any number of scenes per photo can be chosen. clear=true drops all."""
    kind = _kind(kind)
    job = _get(kind, folder)
    if not 0 <= body.index < len(job["items"]):
        raise HTTPException(400, "Unknown photo.")
    it, lib = job["items"][body.index], LIBS[kind]
    if it.get("running"):
        raise HTTPException(409, "This photo is still working.")
    if body.clear:
        with LOCK:
            if kind in R2LIKE:
                for sel in it["sels"]:
                    _drop_sel(job, it, sel, lib)
            it["sels"] = []
            _save(job)
        return {"ok": True}
    cur = _cur_id(it)
    if not cur:
        raise HTTPException(400, "Run step 1 for this photo first.")
    if kind == "r3":
        s = lib.get(body.lib_id or "")
        if not s:
            raise HTTPException(400, "Unknown scene.")
        key = f"{folder}/{it['input']}"
        if s.get("status") == "used" and s.get("photo_key") != key:
            raise HTTPException(400, "That scene was already used for another photo.")
        with LOCK:
            rest = [x for x in it["sels"] if x.get("lib_id") != s["id"]]
            if len(rest) == len(it["sels"]):
                rest.append({"lib_id": s["id"]})
            it["sels"] = rest
            _save(job)
        return {"ok": True}
    run = next((r for r in it["scene_runs"] if r["n"] == body.run), None)
    if not run or body.tile is None or not 0 <= body.tile < len(run["tiles"]):
        raise HTTPException(400, "Unknown scene.")
    with LOCK:
        hit = next((x for x in it["sels"] if x.get("run") == body.run and x.get("tile") == body.tile), None)
        if hit:                                                        # un-choose
            _drop_sel(job, it, hit, lib)
            it["sels"] = [x for x in it["sels"] if x is not hit]
        else:
            lid = lib.add(run["category"], run["subcategory"], {"status": "selected", "text": run["tiles"][body.tile], "photo": it["orig_name"],
                                                              "folder": folder, "text_model": run["text_model"], "effort": run["effort"], "run_n": run["n"]})
            it["sels"].append({"run": body.run, "tile": body.tile, "lib_id": lid})
        _save(job)
    return {"ok": True}


@router.post("/api/rf/{kind}/{folder}/image")
def rf_image(kind: str, folder: str, prompt_b: str = Form(""), size: str = Form("1088x1360"), indexes: str = Form(""), image_choice: str = Form("")):
    job = _get(_kind(kind), folder)
    model, quality = IMAGE_MODEL, QUALITY
    if kind == "r4":
        prompt_b = ""                                     # R4 has no prompt box: the final prompt is the chosen scene alone
        if image_choice not in R4_CHOICES:
            raise HTTPException(400, "Choose the image model and quality.")
        model, quality = R4_CHOICES[image_choice]
    if len(prompt_b) > 30000:
        raise HTTPException(400, "The prompt is too long (limit 30,000 characters).")
    core._check_size(size)
    todo = [i for i in _targets(job, indexes) if job["items"][i].get("sels")]
    if not todo:
        raise HTTPException(400, "Choose a scene (tile) for at least one photo first.")
    _begin(job, todo, "image")
    tasks = []
    with LOCK:
        for i in todo:
            it = job["items"][i]
            gen = max([r.get("gen", 0) for r in it["images"]] + [it.get("gen", 0)]) + 1
            it["gen"], sels = gen, list(it["sels"])
            it["pending"] = len(sels)
            tasks += [(i, sel, gen, k, len(sels)) for k, sel in enumerate(sels, start=1)]
    threading.Thread(target=_run_images, args=(job, tasks, prompt_b.strip(), size, model, quality), daemon=True).start()
    return {"started": len(tasks)}


# ---------------------------------------------------------------- History
def list_rows() -> list:
    rows = []
    for p in core.OUTPUT_DIR.iterdir():
        if not (p.is_dir() and _FOLDER.match(p.name)):
            continue
        try:
            job = _get(p.name.split("_")[0], p.name)
        except HTTPException:
            continue
        items = job["items"]
        imgs = [r for it in items for r in it["images"]]
        try:
            ts = datetime.strptime(job["created"], "%d %b %Y, %H:%M").timestamp()
        except Exception:
            ts = p.stat().st_mtime
        thumb = next((f"/outputs/{p.name}/{it['images'][-1]['file']}" for it in items if it["images"]), "") or (f"/outputs/{p.name}/{items[0]['input']}" if items else "")
        cats = list(dict.fromkeys(f"{r['category']} > {r['subcategory']}" for it in items for r in it["id_runs"] if not r.get("error") and r.get("category")))
        models = [f"{r['text_model']} ({r['effort']})" for it in items for r in it["id_runs"] + it["scene_runs"]] + list(dict.fromkeys(f"{r.get('image_model', IMAGE_MODEL)} {r.get('quality', QUALITY)}" for it in items for r in it["images"])) * (job["kind"] == "r4") + [IMAGE_MODEL] * (job["kind"] != "r4")
        rows.append({
            "type": job["kind"], "flow": job["kind"], "folder": p.name, "created": job["created"], "ts": ts,
            "title": job.get("name") or job["kind"].upper(), "state": "running" if any(it.get("running") for it in items) else "done",
            "thumb": thumb, "photos": len(items), "id_runs": sum(len(it["id_runs"]) for it in items),
            "scene_runs": sum(len(it["scene_runs"]) for it in items), "images_done": len(imgs), "images_total": len(imgs),
            "failed": sum(1 for it in items if it.get("error")), "cost_inr": round(job.get("total_inr", 0), 4),
            "models": list(dict.fromkeys(models)), "categories": cats,
            "selected": [{"photo": it["orig_name"], "text": t} for it in items
                         for t in list(dict.fromkeys(r["scene_text"] for r in it["images"]))][:8],
        })
    return rows
