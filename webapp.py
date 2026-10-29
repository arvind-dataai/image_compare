"""Small web UI + API on top of the existing backend.

  GET  /                      the page (web/index.html)
  GET  /api/options           text/image models, efforts, qualities, rough price estimates
  POST /api/compare           image + prompt + (text models x efforts) + (image models x qualities)
                              -> starts a run, returns {folder}
  GET  /api/runs/{folder}     live status of one run (poll this while it is generating)
  GET  /api/runs/{folder}/zip everything in that run's folder

One run is a small tree, all saved in one numbered folder with a shareable comparison.html:

  image + prompt -> text model + effort -> analysis + written prompt -> image model + quality -> image

Every selected text model x effort is one "branch": it analyses the photo once and writes one image
prompt, and that prompt is then sent to every selected image model x quality. "Direct" is an optional
extra branch that sends the user's own prompt straight to the image models (no text model).

Optional password: set APP_PASSWORD in .env and the browser asks for it (user name can be anything).
"""
import base64
import hashlib
import io
import json
from datetime import datetime
import os
import re
import secrets
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

import main as core
from costing import (ESTIMATE_1024, META_MODELS, NANO_MODELS, TEXT_RATES_PER_M, USD_TO_INR, _as_dict, estimate_inr, text_rate,
                     log_text_cost)
from prompts import build_custom_prompt
from runs import new_run_dir, short_model, write_flow_comparison

router = APIRouter()

# What the page offers. To add a model or quality, add it here (and to the enums in main.py).
WEB_MODELS = {
    "gpt-image-2.5-flare": "Flare: fast, good for everyday edits",
    "gpt-image-2.5-sunburst": "Sunburst: most capable, precise edits",
    "gemini-nano-banana-2.1": "Nano Banana 2.1 (Google): low = 1K, medium = 2K, high = 4K",
    "muse-image-1.0": "Muse Image (Meta): flat $0.01 per image, no quality levels",
}
NO_QUALITY = {"muse-image-1.0"}          # models with no quality setting: one image per branch, labelled "standard"
STANDARD = "standard"
WEB_QUALITIES = ["low", "medium", "high"]
SIZES = {
    "1024x1024": "Square 1:1 (1024x1024)",
    "1088x1360": "Feed 4:5 (1088x1360)",
    "1024x1536": "Portrait 2:3 (1024x1536)",
    "1536x1024": "Landscape 3:2 (1536x1024)",
}

# Text models and the reasoning-effort values each one accepts (from OpenAI's model pages).
TEXT_MODELS = {
    "gpt-6.1-sol": {"slug": "sol", "label": "Sol: balanced",
                    "efforts": ["low", "medium", "high", "xhigh", "max"]},
    "gpt-6-astra": {"slug": "astra", "label": "Astra: most capable, costs more",
                    "efforts": ["low", "medium", "high", "xhigh", "max"]},
    "gpt-6-luna": {"slug": "luna", "label": "Luna: cheapest",
                   "efforts": ["none", "low", "medium", "high", "xhigh", "max"]},
    # Google: thinking_level low | medium (default) | high.  Needs GEMINI_API_KEY.
    "gemini-4-argon": {"slug": "argon", "provider": "google", "label": "Gemini 4 Argon (Google): latest flagship",
                       "efforts": ["low", "medium", "high"]},
    "gemini-3.8-flash": {"slug": "flash", "provider": "google", "label": "Gemini 3.8 Flash (Google): fast, low price",
                         "efforts": ["low", "medium", "high"]},
    # Meta Model API: reasoning effort minimal..max ("none" is refused).  Needs MODEL_API_KEY.
    "muse-spark-1.3": {"slug": "spark", "provider": "meta", "label": "Muse Spark 1.3 (Meta): fast, low price",
                       "efforts": ["minimal", "low", "medium", "high", "xhigh", "max"]},
    "muse-wonder-1.0": {"slug": "wonder", "provider": "meta", "label": "Muse Wonder 1.0 (Meta): most capable, costs more",
                        "efforts": ["minimal", "low", "medium", "high", "xhigh", "max"]},
}
TEXT_PROVIDER = {k: v.get("provider", "openai") for k, v in TEXT_MODELS.items()}
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
# Rough tokens for the price preview only (output includes hidden reasoning). Real cost is shown after the run.
EST_TEXT_IN = 3500
EST_TEXT_OUT = {"none": 900, "minimal": 1200, "low": 1800, "medium": 3500, "high": 7000, "xhigh": 14000, "max": 24000}
MAX_IMAGES = int(os.getenv("WEB_MAX_IMAGES", "15"))     # safety cap on one run

# The instructions the text model follows. Edit optimizer_prompt.txt to change them.
OPTIMIZER_INSTRUCTIONS = (Path(__file__).parent / "optimizer_prompt.txt").read_text(encoding="utf-8")

POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WEB_WORKERS", "4")))              # image calls
TEXT_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WEB_TEXT_WORKERS", "3")))    # one worker per branch
_COUNT_LOCK = threading.Lock()
JOBS: dict = {}          # folder -> live status of runs started since the server booted
_FOLDER = re.compile(r"^\d+_[A-Za-z0-9_.+\-]+$")


# ---------------------------------------------------------------- password (optional)
class PasswordMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        pw = os.getenv("APP_PASSWORD", "")
        if not pw or request.url.path == "/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                given = base64.b64decode(header[6:]).decode("utf-8", "ignore").partition(":")[2]
            except Exception:
                given = ""
            if secrets.compare_digest(given.encode(), pw.encode()):
                return await call_next(request)
        return Response("Password required", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="Image compare"'})


# ---------------------------------------------------------------- helpers
def _run_json(folder: str) -> dict | None:
    p = core.OUTPUT_DIR / folder / "run.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _check_folder(folder: str) -> Path:
    if not _FOLDER.match(folder) or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "Run not found.")
    return core.OUTPUT_DIR / folder


# ---------------------------------------------------------------- the background job
_FINAL_RE = re.compile(r"^[#*>\s_-]*FINAL\s+PROMPT[ \t*#:_-]*(.*)$", re.IGNORECASE | re.MULTILINE)


def split_reply(text: str) -> tuple[str, str]:
    """Split the text model's reply into (analysis, final prompt). Raises ValueError if there is no prompt."""
    hits = list(_FINAL_RE.finditer(text))
    if not hits:
        raise ValueError("the reply has no 'FINAL PROMPT' section")
    m = hits[-1]
    same_line = m.group(1).strip()
    final = (same_line + "\n" + text[m.end():].strip()).strip() if same_line else text[m.end():].strip()
    final = final.strip().strip('"\u201c\u201d').strip()
    analysis = re.sub(r"^[#*>\s_-]*ANALYSIS[ \t*#:_-]*\n?", "", text[:m.start()].strip(), flags=re.IGNORECASE).strip()
    if len(final) < 40:
        raise ValueError("the final prompt looks too short")
    return analysis, final


def _bump(job: dict, n: int = 1):
    with _COUNT_LOCK:
        job["done"] += n


def _text_step(branch: dict, img, prompt: str, folder: str, ledger: list):
    """One text-model call: reads the photo + the user's request, returns (analysis, final prompt)."""
    text, usage = core.generate_text(TEXT_PROVIDER[branch["text_model"]], branch["text_model"], branch["effort"],
                                     OPTIMIZER_INSTRUCTIONS, f"User request: {prompt}", img)
    cost, source, _ = log_text_cost(core.COST_LOG, branch["text_model"], usage,
                                    note=f"web analysis, effort {branch['effort']} | {folder}")
    u = _as_dict(usage)
    ledger.append({"part": f"Analysis: {branch['label']}", "model": branch["text_model"],
                   "tokens_in": int(u.get("input_tokens") or 0), "tokens_out": int(u.get("output_tokens") or 0),
                   "cost_inr": cost, "source": source})
    branch["text_cost_inr"] = round(cost, 4)
    branch["analysis_raw"] = text
    return split_reply(text)


def _run_branch(job: dict, branch: dict, folder: str, run_dir: Path, img, prompt: str, size: str, ledger: list):
    """text step (unless direct) -> image step for every image model x quality."""
    items = branch["items"]
    if branch["kind"] == "text":
        branch["status"] = "running"
        try:
            branch["analysis"], branch["final_prompt"] = _text_step(branch, img, prompt, folder, ledger)
            branch.pop("analysis_raw", None)
        except Exception as e:
            detail = getattr(e, "detail", None) or str(e)
            if branch.get("analysis_raw"):                      # keep what the model wrote, it is still useful
                branch["analysis"] = branch["analysis_raw"]
            branch.update(status="error", error=f"Text step failed: {detail}")
            for it in items:
                it.update(status="skipped", error="Skipped because the text step failed.")
            _bump(job, 1 + len(items))
            return
        _bump(job)
    else:
        branch["status"] = "running"
        branch["final_prompt"] = build_custom_prompt(prompt)    # your words + the standard preservation rules
    image_prompt = branch["final_prompt"]

    def one(it):
        it["status"] = "running"
        try:
            png, usage = core._generate(img, image_prompt, size, it["model"], it["quality"])
            fname = f"{branch['prefix']}{short_model(it['model'])}_{it['quality']}.png"
            cost = core._store(run_dir, fname, png, usage, size, it["model"], it["quality"],
                               f"web {branch['label']}", ledger)
            it.update(status="done", file=fname, url=f"/outputs/{folder}/{fname}", cost_inr=round(cost, 4))
        except Exception as e:                                  # one failed image must not stop the others
            it.update(status="error", error=str(getattr(e, "detail", None) or e))
        _bump(job)

    list(POOL.map(one, items))
    branch["status"] = "done"


def _run_job(folder: str, run_dir: Path, img, in_name: str, prompt: str, size: str):
    job = JOBS[folder]
    ledger: list = []
    try:
        list(TEXT_POOL.map(lambda b: _run_branch(job, b, folder, run_dir, img, prompt, size, ledger),
                           job["branches"]))
        # the cost table lists text steps first, then images, in a stable order
        ledger.sort(key=lambda c: (not c["part"].startswith("Analysis"), c["part"]))
        total = sum(c["cost_inr"] for c in ledger)
        n_img = sum(len(b["items"]) for b in job["branches"])
        n_text = sum(1 for b in job["branches"] if b["kind"] == "text")
        write_flow_comparison(
            run_dir, f"Compare: {prompt[:70]}", in_name, prompt, job["branches"],
            {"Prompt": prompt, "Size": size, "Text steps": n_text, "Images": n_img,
             "Total cost": f"\u20b9{total:.2f}"},
            costs=ledger,
        )
        job["total_inr"] = round(total, 4)
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)


# ---------------------------------------------------------------- API
@router.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


def _text_ready(model: str) -> bool:
    p = TEXT_PROVIDER[model]
    return core.gemini_ready() if p == "google" else core.meta_ready() if p == "meta" else True


@router.get("/api/options")
def options():
    return {
        "models": [{"id": k, "label": v, "available": core.gemini_ready() if k in NANO_MODELS else core.meta_ready() if k in META_MODELS else True,
                    "needs": "GEMINI_API_KEY in the server's .env" if k in NANO_MODELS else "MODEL_API_KEY in the server's .env" if k in META_MODELS else "",
                    "no_quality": k in NO_QUALITY,
                    # rupees per image for each quality; fixed=True means the price does not depend on the size picked
                    "est_inr": {q: round(estimate_inr(q, k), 2) for q in WEB_QUALITIES},
                    "fixed": k in NANO_MODELS or k in META_MODELS} for k, v in WEB_MODELS.items()],
        "qualities": WEB_QUALITIES,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"],
                         "rate_in": text_rate(k)[0], "rate_out": text_rate(k)[1],
                         "available": _text_ready(k),
                         "needs": ("GEMINI_API_KEY" if TEXT_PROVIDER[k] == "google" else "MODEL_API_KEY" if TEXT_PROVIDER[k] == "meta" else "") + " in the server's .env"}
                        for k, v in TEXT_MODELS.items()],
        "efforts": EFFORTS,
        "usd_to_inr": USD_TO_INR,
        "est_text_in": EST_TEXT_IN,
        "est_text_out": EST_TEXT_OUT,
        "max_images": MAX_IMAGES,
        "max_upload_mb": core.MAX_UPLOAD_MB,
        # rough rupees per 1024x1024 image; the page scales this by pixel count
        "est_inr_1024": {q: round(ESTIMATE_1024[q] * USD_TO_INR, 2) for q in WEB_QUALITIES},
        "password_protected": bool(os.getenv("APP_PASSWORD")),
    }


def _validate(prompts, models, qualities, size, text_models, efforts, direct):
    """Check the options once. Returns the cleaned settings used to launch every run."""
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise HTTPException(400, "Enter a prompt.")
    models = [m for m in WEB_MODELS if m in models]            # keeps a fixed order, drops unknowns
    qualities = [q for q in WEB_QUALITIES if q in qualities]
    if not models or (not qualities and any(m not in NO_QUALITY for m in models)):
        raise HTTPException(400, "Select at least one image model and one quality.")
    if any(m in NANO_MODELS for m in models) and not core.gemini_ready():
        raise HTTPException(400, "Nano Banana needs GEMINI_API_KEY in the server's .env file.")
    if any(m in META_MODELS for m in models) and not core.meta_ready():
        raise HTTPException(400, "Muse Image needs MODEL_API_KEY in the server's .env file.")
    text_models = [t for t in TEXT_MODELS if t in text_models]
    efforts = [x for x in EFFORTS if x in efforts]
    for t in text_models:
        if not _text_ready(t):
            raise HTTPException(400, f"{t} needs {'GEMINI_API_KEY' if TEXT_PROVIDER[t] == 'google' else 'MODEL_API_KEY'} in the server's .env file.")
    if text_models and not efforts:
        raise HTTPException(400, "Select at least one effort level for the text models.")
    plans = [(t, x) for t in text_models for x in efforts if x in TEXT_MODELS[t]["efforts"]]
    if text_models and not plans:
        raise HTTPException(400, "None of the selected effort levels is supported by the selected text models.")
    use_direct = direct or not plans                            # nothing chosen -> plain prompt, as before
    core._check_size(size)

    n_branches = len(plans) + (1 if use_direct else 0)
    combos = [(m, q) for m in models for q in ([STANDARD] if m in NO_QUALITY else qualities)]
    n_images = n_branches * len(combos)
    if n_images > MAX_IMAGES:
        raise HTTPException(400, f"That would make {n_images} images per prompt and photo (limit {MAX_IMAGES}). Select fewer options.")
    return dict(prompts=prompts, models=models, size=size, plans=plans, use_direct=use_direct,
                combos=combos, n_images=n_images)


def _launch(img, prompt: str, cfg: dict) -> str:
    """Create one run (one photo + one prompt) and start it in a background thread. Returns its folder."""
    plans, use_direct, combos, models, size = cfg["plans"], cfg["use_direct"], cfg["combos"], cfg["models"], cfg["size"]
    branches = []
    if use_direct:
        branches.append({"id": "direct", "kind": "direct", "slug": "direct",
                         "label": "Direct: your prompt, no text model"})
    for t, x in plans:
        branches.append({"id": f"{TEXT_MODELS[t]['slug']}-{x}", "kind": "text", "slug": f"{TEXT_MODELS[t]['slug']}-{x}",
                         "text_model": t, "effort": x, "label": f"{t} \u00b7 {x}"})
    only_direct = not plans
    for b in branches:
        b["status"] = "queued"
        b["prefix"] = "" if only_direct else f"{b['slug']}__"
        b["items"] = [{"label": f"{short_model(m)} \u00b7 {q}", "model": m, "quality": q, "status": "queued"}
                      for m, q in combos]

    gpt = [short_model(m).replace("gpt2.5", "") for m in models if m.startswith("gpt-image-2.5")]
    other = [short_model(m) for m in models if not m.startswith("gpt-image-2.5")]
    mpart = "+".join(([("gpt2.5" + "+".join(gpt))] if gpt else []) + other)
    if not only_direct:
        slugs = (["direct"] if use_direct else []) + list(dict.fromkeys(TEXT_MODELS[t]["slug"] for t, _ in plans))
        mpart = "+".join(slugs) + "__" + mpart
    run_dir = new_run_dir(core.OUTPUT_DIR, mpart, "+".join(dict.fromkeys(q for _, q in combos)))
    folder = run_dir.name
    in_name = core._save_input(run_dir, img)
    JOBS[folder] = {
        "folder": folder, "state": "running", "prompt": prompt, "size": size,
        "total": len(plans) + cfg["n_images"], "done": 0,
        "input_url": f"/outputs/{folder}/{in_name}", "branches": branches,
        "image_key": hashlib.md5(img[1]).hexdigest(),
    }
    threading.Thread(target=_run_job, args=(folder, run_dir, img, in_name, prompt, size), daemon=True).start()
    return folder


@router.post("/api/compare")
async def compare(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    models: list[str] = Form(...),
    qualities: list[str] = Form([]),
    size: str = Form("1024x1024"),
    text_models: list[str] = Form([]),
    efforts: list[str] = Form([]),
    direct: bool = Form(False),
):
    cfg = _validate([prompt], models, qualities, size, text_models, efforts, direct)
    img = await core._read_image(image)
    folder = _launch(img, cfg["prompts"][0], cfg)
    return {"folder": folder, "images": cfg["n_images"], "text_steps": len(cfg["plans"])}


# ---- several prompts x several photos: prompt 1 on every photo, then prompt 2 on every photo, ...
GROUPS: dict = {}
GROUPS_FILE = Path(__file__).parent / "run_groups.json"       # kept on disk so a group can be reopened later from History
_GLOCK = threading.Lock()
MAX_PAIRS = int(os.getenv("WEB_MAX_PAIRS", "30"))             # safety cap: prompts x photos in one go


def _save_groups():
    try:
        with _GLOCK:
            GROUPS_FILE.write_text(json.dumps(GROUPS, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass                                                    # saving the list is best-effort


def _load_groups():
    try:
        data = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for gid, g in data.items():
        if g.get("state") == "running":                        # the server restarted while it was running
            g["state"] = "done"
            for e in g["entries"]:
                if e["state"] in ("queued", "running"):
                    e["state"], e["error"] = "error", "Stopped: the server restarted before this run finished."
        GROUPS[gid] = g


_load_groups()


def _run_group(gid: str, imgs: list, cfg: dict):
    g = GROUPS[gid]
    try:
        for e in g["entries"]:                                  # one pair at a time, in order
            e["state"] = "running"
            _save_groups()
            try:
                e["folder"] = _launch(imgs[e["image_no"] - 1], e["prompt"], cfg)
                e["input_url"] = JOBS[e["folder"]]["input_url"]
                _save_groups()
                job = JOBS[e["folder"]]
                while job.get("state") == "running":
                    time.sleep(1.5)
                e["state"] = "error" if job.get("state") == "error" else "done"
                e["error"] = job.get("error", "")
            except Exception as ex:                             # a failed pair must not stop the others
                e["state"] = "error"
                e["error"] = str(getattr(ex, "detail", None) or ex)
            _save_groups()
    finally:
        g["state"] = "done"
        _save_groups()


@router.post("/api/compare_multi")
async def compare_multi(
    images: list[UploadFile] = File(...),
    prompts: list[str] = Form(...),
    models: list[str] = Form(...),
    qualities: list[str] = Form([]),
    size: str = Form("1024x1024"),
    text_models: list[str] = Form([]),
    efforts: list[str] = Form([]),
    direct: bool = Form(False),
):
    cfg = _validate(prompts, models, qualities, size, text_models, efforts, direct)
    if not images:
        raise HTTPException(400, "Add at least one product image.")
    pairs = len(cfg["prompts"]) * len(images)
    if pairs > MAX_PAIRS:
        raise HTTPException(400, f"That would be {pairs} runs (limit {MAX_PAIRS}). Use fewer prompts or photos.")
    imgs = [await core._read_image(f) for f in images]
    names = [f.filename or f"image {i + 1}" for i, f in enumerate(images)]
    gid = f"g{int(time.time())}{secrets.token_hex(2)}"
    entries = [{"prompt_no": pi + 1, "image_no": ii + 1, "prompt": p, "image_name": names[ii],
                "state": "queued", "folder": "", "input_url": "", "error": ""}
               for pi, p in enumerate(cfg["prompts"]) for ii in range(len(imgs))]
    GROUPS[gid] = {"id": gid, "state": "running", "entries": entries, "prompts": cfg["prompts"], "image_names": names,
                   "size": size, "created": datetime.now().strftime("%d %b %Y, %H:%M")}
    _save_groups()
    threading.Thread(target=_run_group, args=(gid, imgs, cfg), daemon=True).start()
    return {"group": gid, "runs": pairs, "images_per_run": cfg["n_images"], "text_steps_per_run": len(cfg["plans"])}


@router.get("/api/groups/{gid}")
def group_status(gid: str):
    g = GROUPS.get(gid)
    if not g:
        raise HTTPException(404, "Unknown group (the server may have restarted). Open the runs from History.")
    return g


_KEYS: dict = {}


def _image_key(folder: str, name: str) -> str:
    """Fingerprint of a run's input photo, so runs on the same photo can be put side by side (also for old runs)."""
    k = (folder, name)
    if k not in _KEYS:
        try:
            _KEYS[k] = hashlib.md5((core.OUTPUT_DIR / folder / name).read_bytes()).hexdigest()
        except Exception:
            _KEYS[k] = ""
    return _KEYS[k]


@router.get("/api/runs/{folder}")
def run_status(folder: str):
    _check_folder(folder)
    if folder in JOBS:
        return JOBS[folder]
    d = _run_json(folder)
    if not d:
        return {"folder": folder, "state": "unknown"}

    def with_urls(items):
        return [{**it, "status": "done" if it.get("file") else it.get("status", "error"),
                 "url": f"/outputs/{folder}/{it['file']}" if it.get("file") else None} for it in items]

    branches = d.get("branches")
    if branches:
        branches = [{**b, "items": with_urls(b["items"])} for b in branches]
    else:                                   # runs saved before text models existed: one direct branch
        branches = [{"id": "direct", "kind": "direct", "label": "Direct: your prompt, no text model",
                     "status": "done", "items": with_urls(d["items"])}]
    return {"folder": folder, "state": "done", "total_inr": d["total_inr"], "prompt": d["meta"].get("Prompt", ""),
            "size": d["meta"].get("Size", "1024x1024"),
            "input_url": f"/outputs/{folder}/{d['input']}", "image_key": _image_key(folder, d["input"]),
            "branches": branches}


def _ts(text: str, fallback: float = 0.0) -> float:
    try:
        return datetime.strptime(text, "%d %b %Y, %H:%M").timestamp()
    except Exception:
        return fallback


@router.get("/api/history")
def history():
    """Every run in one list, newest first: single runs from the New run tab and batch runs (scene flow or older direct batches)."""
    import batch                                  # imported here because batch imports main, which imports this module
    rows = []
    for p in core.OUTPUT_DIR.iterdir():
        if not (p.is_dir() and re.match(r"^\d+_", p.name)):
            continue
        d, job = _run_json(p.name), JOBS.get(p.name)
        if d is None and not job:
            continue
        mtime = p.stat().st_mtime
        when = (d or {}).get("time") or datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
        if d is not None:
            branches = d.get("branches") or [{"items": d.get("items", [])}]
            items = [it for b in branches for it in b.get("items", [])]
            texts = [f"{b['text_model']} ({b.get('effort', '')})" for b in branches if b.get("text_model")]
            models = list(dict.fromkeys(f"{it['model']} {it.get('quality') or ''}".strip() for it in items if it.get("model")))
            state = "running" if job and job.get("state") == "running" else "done"
            rows.append({
                "type": "single", "flow": None, "folder": p.name, "created": when, "ts": _ts(when, mtime),
                "title": (d.get("meta") or {}).get("Prompt") or d.get("title", ""), "state": state,
                "thumb": f"/outputs/{p.name}/{d['input']}" if d.get("input") else "",
                "images_total": len(items), "images_done": sum(1 for it in items if it.get("file")),
                "failed": sum(1 for it in items if not it.get("file") and it.get("error")),
                "cost_inr": d.get("total_inr", 0), "models": texts + models, "categories": []})
        else:                                     # started since the server booted, not finished yet
            rows.append({
                "type": "single", "flow": None, "folder": p.name, "created": when, "ts": mtime,
                "title": job.get("prompt", ""), "state": job.get("state", "running"), "thumb": "",
                "images_total": 0, "images_done": 0, "failed": 0, "cost_inr": job.get("total_inr", 0),
                "models": [], "categories": []})
    for g in GROUPS.values():
        ents = g["entries"]
        cost = sum(((_run_json(e["folder"]) or {}).get("total_inr") or (JOBS.get(e["folder"]) or {}).get("total_inr") or 0)
                   for e in ents if e.get("folder"))
        first = next((e["input_url"] for e in ents if e.get("input_url")), "")
        when = g.get("created", "")
        rows.append({
            "type": "multi", "flow": None, "folder": g["id"], "created": when, "ts": _ts(when, 0.0),
            "title": f"{len(g['prompts'])} prompt{'s' if len(g['prompts']) != 1 else ''} \u00d7 {len(g['image_names'])} image"
                     f"{'s' if len(g['image_names']) != 1 else ''}: " + " | ".join(p[:50] for p in g["prompts"]),
            "state": g.get("state", "done"), "thumb": first,
            "images_total": len(ents), "images_done": sum(1 for e in ents if e["state"] == "done"),
            "failed": sum(1 for e in ents if e["state"] == "error"),
            "cost_inr": round(cost, 4), "models": [], "categories": []})
    import r1
    import rflow
    rows.extend(r1.r1_list())
    rows.extend(rflow.list_rows())
    for b in batch.batch_list()["batches"]:
        rows.append({
            "type": "batch", "flow": b["flow"], "folder": b["folder"], "created": b["created"],
            "ts": _ts(b["created"], 0.0), "title": b["name"] or b["prompt"], "state": b["state"],
            "thumb": b.get("thumb", ""), "images_total": b["total"], "images_done": b["done"], "failed": b["failed"],
            "cost_inr": b["total_inr"], "models": [], "categories": b["categories"],
            "products": b.get("products", 0), "scene_runs": b.get("scene_runs", 0), "image_runs": b.get("image_runs", 0)})
    rows.sort(key=lambda r: -r["ts"])
    return {"runs": rows}


@router.get("/api/runs/{folder}/zip")
def run_zip(folder: str):
    run_dir = _check_folder(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.iterdir()):
            if f.is_file():
                z.write(f, f"{folder}/{f.name}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{folder}.zip"'})


# ---------------------------------------------------------------- Excel export
def _png_size(path: Path):
    """(width, height) of a PNG/JPEG without Pillow; None if unknown."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                import struct
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"\xff\xd8":                       # JPEG: walk the segments
                import struct
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    m = f.read(1)
                    while m == b"\xff":
                        m = f.read(1)
                    if not m:
                        return None
                    if m[0] in (0xC0, 0xC1, 0xC2):
                        f.read(3)
                        h, w = struct.unpack(">HH", f.read(4))
                        return w, h
                    ln = struct.unpack(">H", f.read(2))[0]
                    f.seek(ln - 2, 1)
    except Exception:
        return None
    return None


def _public_base(request) -> str:
    env = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
    if env:
        return env
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or ("http" if host.startswith(("127.", "localhost")) else "https")
    return f"{proto}://{host}" if host else ""
