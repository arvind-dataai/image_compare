"""Image tab: photo(s) + your prompt -> image, with Flare (low) or Sunburst (low). Nothing else is added to your prompt.

  GET  /api/ig/options          image models (flare low, sunburst low), sizes
  POST /api/ig/run              photos + prompt + model + size -> starts the images, returns {folder}
  GET  /api/ig/{folder}         the whole run
  POST /api/ig/{folder}/image   make another image (new prompt / model / size allowed), for all photos or the chosen ones

Photos can be made one image each ("separate") or all sent together as references for ONE image ("together").
Stored in outputs/ig_<n>_<name>/ig.json and listed in History.
"""
import copy
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import batch
import imgchoices
import main as core

router = APIRouter()
IMAGE_CHOICES = ("flare-low", "sunburst-low")
_FOLDER = re.compile(r"^ig_\d+_[A-Za-z0-9_.+\-]+$")
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=int(__import__("os").getenv("WEB_BATCH_WORKERS", "3")))
JOBS: dict = {}
MAX_PHOTOS = 20


def _now() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _file(folder: str) -> Path:
    return core.OUTPUT_DIR / folder / "ig.json"


def _save(job: dict):
    p = _file(job["folder"])
    tmp = p.with_suffix(".tmp")
    with LOCK:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def _get(folder: str) -> dict:
    if not _FOLDER.match(folder) or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "Image run not found.")
    with LOCK:
        if folder in JOBS:
            return JOBS[folder]
        try:
            job = json.loads(_file(folder).read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(404, "Image run not found.")
        for it in job["items"]:
            if it.get("running"):
                it["running"], it["error"] = False, "Stopped: the server restarted before this finished."
        JOBS[folder] = job
        return job


def _new_folder(name: str) -> Path:
    with LOCK:
        nums = [int(p.name.split("_")[1]) for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and _FOLDER.match(p.name)]
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:24] or "image"
        d = core.OUTPUT_DIR / f"ig_{max(nums, default=0) + 1}_{slug}"
        d.mkdir()
    return d


def _view(job: dict) -> dict:
    with LOCK:
        out = copy.deepcopy(job)
    f = job["folder"]
    for it in out["items"]:
        it["input_urls"] = [f"/outputs/{f}/{n}" for n in it["inputs"]]
        for r in it["images"]:
            r["url"] = f"/outputs/{f}/{r['file']}"
    return out


def _images_of(folder: str, it: dict):
    out = []
    for n in it["inputs"]:
        ctype = "image/png" if n.lower().endswith(".png") else "image/webp" if n.lower().endswith(".webp") else "image/jpeg"
        out.append((n, (core.OUTPUT_DIR / folder / n).read_bytes(), ctype))
    return out[0] if len(out) == 1 else out


def _do(job: dict, i: int, prompt: str, size: str, choice: str):
    it, folder = job["items"][i], job["folder"]
    err = ""
    try:
        model, quality = imgchoices.pick(choice)
        with LOCK:
            n = max(it.get("seq", 0), len(it["images"])) + 1
            it["seq"] = n
        png, usage = core._generate(_images_of(folder, it), prompt, size, model, quality)
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}_{n}.png"
        cost = core._store(core.OUTPUT_DIR / folder, fname, png, usage, size, model, quality, f"image tab | {folder}")
        with LOCK:
            it["images"].append({"n": n, "created": _now(), "prompt": prompt, "size": size, "image_model": model, "quality": quality,
                                 "choice": choice, "file": fname, "cost_inr": round(cost, 4),
                                 "usage": usage if isinstance(usage, dict) else core._as_dict(usage)})
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
            job["size"], job["prompt"], job["choice"] = size, prompt, choice
    except Exception as e:
        err = str(getattr(e, "detail", None) or e)
    with LOCK:
        it.update(running=False, error=err)
        _save(job)


def _run(job: dict, todo: list, prompt: str, size: str, choice: str):
    try:
        list(POOL.map(lambda i: _do(job, i, prompt, size, choice), todo))
    except Exception as e:
        with LOCK:
            for i in todo:
                if job["items"][i].get("running"):
                    job["items"][i].update(running=False, error=str(e))
            _save(job)


def _start(job: dict, todo: list, prompt: str, size: str, choice: str):
    with LOCK:
        busy = [i + 1 for i in todo if job["items"][i].get("running")]
        if busy:
            raise HTTPException(409, f"Photo {', '.join(map(str, busy))} is still working. Wait for it to finish.")
        for i in todo:
            job["items"][i].update(running=True, error="")
        _save(job)
    threading.Thread(target=_run, args=(job, todo, prompt, size, choice), daemon=True).start()


def _check(prompt: str, size: str, choice: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Write the image prompt.")
    if len(prompt) > 30000:
        raise HTTPException(400, "The prompt is too long (limit 30,000 characters).")
    if choice not in IMAGE_CHOICES:
        raise HTTPException(400, "Choose the image model.")
    core._check_size(size)
    return prompt


@router.get("/api/ig/options")
def ig_options():
    from webapp import SIZES
    return {"sizes": [{"id": k, "label": v} for k, v in SIZES.items()], "max_photos": MAX_PHOTOS,
            "image_choices": [c for c in imgchoices.for_page() if c["id"] in IMAGE_CHOICES]}


@router.post("/api/ig/run")
async def ig_run(images: list[UploadFile] = File(...), prompt: str = Form(...), size: str = Form("1088x1360"),
                 image_choice: str = Form(...), mode: str = Form("separate"), name: str = Form("")):
    prompt = _check(prompt, size, image_choice)
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > MAX_PHOTOS:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {MAX_PHOTOS} per run.")
    read = [await core._read_image(up) for up in images]
    d = _new_folder(name or "image")
    files = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        files.append((fname, img[0]))
    together = mode == "together" and len(files) > 1
    groups = [files] if together else [[f] for f in files]
    items = [{"inputs": [f[0] for f in g], "orig_name": (g[0][1] if len(g) == 1 else f"{len(g)} photos together"), "images": [],
              "running": False, "error": ""} for g in groups]
    job = {"folder": d.name, "flow": "ig", "name": name.strip() or "Image", "created": _now(), "mode": "together" if together else "separate",
           "size": size, "prompt": prompt, "choice": image_choice, "total_inr": 0, "items": items}
    with LOCK:
        JOBS[d.name] = job
    _save(job)
    _start(job, list(range(len(items))), prompt, size, image_choice)
    return {"folder": d.name, "items": len(items)}


@router.get("/api/ig/{folder}")
def ig_get(folder: str):
    return _view(_get(folder))


@router.post("/api/ig/{folder}/image")
def ig_image(folder: str, prompt: str = Form(...), size: str = Form("1088x1360"), image_choice: str = Form(...), indexes: str = Form("")):
    job = _get(folder)
    prompt = _check(prompt, size, image_choice)
    n = len(job["items"])
    try:
        todo = sorted({int(x) for x in indexes.split(",") if x.strip() != ""}) if indexes.strip() else list(range(n))
    except ValueError:
        raise HTTPException(400, "Bad photo list.")
    if not todo or any(i < 0 or i >= n for i in todo):
        raise HTTPException(400, "Unknown photo.")
    _start(job, todo, prompt, size, image_choice)
    return {"started": len(todo)}


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
            or (f"/outputs/{p.name}/{items[0]['inputs'][0]}" if items else "")
        rows.append({
            "type": "ig", "flow": "ig", "folder": p.name, "created": job["created"], "ts": ts, "title": job.get("name") or "Image",
            "state": "running" if any(it.get("running") for it in items) else "done", "thumb": thumb,
            "photos": len(items), "images_done": len(imgs), "images_total": len(imgs),
            "failed": sum(1 for it in items if it.get("error")), "cost_inr": round(job.get("total_inr", 0), 4),
            "models": list(dict.fromkeys(f"{r['image_model']} {r['quality']}" for r in imgs)),
            "categories": [], "selected": [{"photo": it["orig_name"], "text": r["prompt"]} for it in items for r in it["images"][-1:]][:8],
        })
    return rows
