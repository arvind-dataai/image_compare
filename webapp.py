"""Small web UI + API on top of the existing backend.

  GET  /                      the page (web/index.html)
  GET  /api/options           text/image models, efforts, qualities, rough price estimates
  POST /api/compare           image + prompt + (text models x efforts) + (image models x qualities)
                              -> starts a run, returns {folder}
  GET  /api/runs              history of all runs
  GET  /api/runs/{folder}     live status of one run (poll this while it is generating)
  GET  /api/runs/{folder}/zip everything in that run's folder
  GET  /api/costlog           cost_log.csv as JSON rows (for the Cost log tab)
  GET  /cost.csv              download cost_log.csv

One run is a small tree, all saved in one numbered folder with a shareable comparison.html:

  image + prompt -> text model + effort -> analysis + written prompt -> image model + quality -> image

Every selected text model x effort is one "branch": it analyses the photo once and writes one image
prompt, and that prompt is then sent to every selected image model x quality. "Direct" is an optional
extra branch that sends the user's own prompt straight to the image models (no text model).

Optional password: set APP_PASSWORD in .env and the browser asks for it (user name can be anything).
"""
import base64
import csv
import io
import json
from datetime import datetime
import os
import re
import secrets
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

import main as core
from costing import (ESTIMATE_1024, LOG_FIELDS, NANO_MODELS, TEXT_RATES_PER_M, USD_TO_INR, _as_dict, estimate_inr,
                     log_text_cost, total_so_far)
from prompts import build_custom_prompt
from runs import new_run_dir, short_model, write_flow_comparison

router = APIRouter()

# What the page offers. To add a model or quality, add it here (and to the enums in main.py).
WEB_MODELS = {
    "gpt-image-2.5-flare": "Flare: fast, good for everyday edits",
    "gpt-image-2.5-sunburst": "Sunburst: most capable, precise edits",
    "gemini-nano-banana-2.1": "Nano Banana 2.1 (Google): low = 1K, medium = 2K, high = 4K",
}
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
}
EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]
# Rough tokens for the price preview only (output includes hidden reasoning). Real cost is shown after the run.
EST_TEXT_IN = 3500
EST_TEXT_OUT = {"none": 900, "low": 1800, "medium": 3500, "high": 7000, "xhigh": 14000, "max": 24000}
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


def _cost_rows() -> list:
    p = Path(core.COST_LOG)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    b64 = base64.b64encode(img[1]).decode()
    resp = core.client.responses.create(
        model=branch["text_model"],
        instructions=OPTIMIZER_INSTRUCTIONS,
        reasoning={"effort": branch["effort"]},
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": f"User request: {prompt}"},
            {"type": "input_image", "image_url": f"data:{img[2]};base64,{b64}"},
        ]}],
    )
    text = resp.output_text or ""
    usage = getattr(resp, "usage", None)
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


@router.get("/api/options")
def options():
    return {
        "models": [{"id": k, "label": v, "available": core.gemini_ready() if k in NANO_MODELS else True,
                    "needs": "GEMINI_API_KEY in the server's .env" if k in NANO_MODELS else "",
                    # rupees per image for each quality; fixed=True means the price does not depend on the size picked
                    "est_inr": {q: round(estimate_inr(q, k), 2) for q in WEB_QUALITIES},
                    "fixed": k in NANO_MODELS} for k, v in WEB_MODELS.items()],
        "qualities": WEB_QUALITIES,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"],
                         "rate_in": TEXT_RATES_PER_M[k][0], "rate_out": TEXT_RATES_PER_M[k][1]}
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


@router.post("/api/compare")
async def compare(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    models: list[str] = Form(...),
    qualities: list[str] = Form(...),
    size: str = Form("1024x1024"),
    text_models: list[str] = Form([]),
    efforts: list[str] = Form([]),
    direct: bool = Form(False),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Enter a prompt.")
    models = [m for m in WEB_MODELS if m in models]            # keeps a fixed order, drops unknowns
    qualities = [q for q in WEB_QUALITIES if q in qualities]
    if not models or not qualities:
        raise HTTPException(400, "Select at least one image model and one quality.")
    if any(m in NANO_MODELS for m in models) and not core.gemini_ready():
        raise HTTPException(400, "Nano Banana needs GEMINI_API_KEY in the server's .env file.")
    text_models = [t for t in TEXT_MODELS if t in text_models]
    efforts = [x for x in EFFORTS if x in efforts]
    if text_models and not efforts:
        raise HTTPException(400, "Select at least one effort level for the text models.")
    plans = [(t, x) for t in text_models for x in efforts if x in TEXT_MODELS[t]["efforts"]]
    if text_models and not plans:
        raise HTTPException(400, "None of the selected effort levels is supported by the selected text models.")
    use_direct = direct or not plans                            # nothing chosen -> plain prompt, as before
    core._check_size(size)

    n_branches = len(plans) + (1 if use_direct else 0)
    n_images = n_branches * len(models) * len(qualities)
    if n_images > MAX_IMAGES:
        raise HTTPException(400, f"That would make {n_images} images (limit {MAX_IMAGES}). Select fewer options.")
    img = await core._read_image(image)

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
                      for m in models for q in qualities]

    gpt = [short_model(m).replace("gpt2.5", "") for m in models if m.startswith("gpt-image-2.5")]
    other = [short_model(m) for m in models if not m.startswith("gpt-image-2.5")]
    mpart = "+".join(([("gpt2.5" + "+".join(gpt))] if gpt else []) + other)
    if not only_direct:
        slugs = (["direct"] if use_direct else []) + list(dict.fromkeys(TEXT_MODELS[t]["slug"] for t, _ in plans))
        mpart = "+".join(slugs) + "__" + mpart
    run_dir = new_run_dir(core.OUTPUT_DIR, mpart, "+".join(qualities))
    folder = run_dir.name
    in_name = core._save_input(run_dir, img)
    JOBS[folder] = {
        "folder": folder, "state": "running", "prompt": prompt, "size": size,
        "total": len(plans) + n_images, "done": 0,
        "input_url": f"/outputs/{folder}/{in_name}", "branches": branches,
    }
    threading.Thread(target=_run_job, args=(folder, run_dir, img, in_name, prompt, size), daemon=True).start()
    return {"folder": folder, "images": n_images, "text_steps": len(plans)}


@router.get("/api/prompts")
def prompts():
    """Distinct prompts used in earlier runs, newest first (for the 'Recent prompts' list)."""
    found = []
    for p in core.OUTPUT_DIR.iterdir():
        mm = re.match(r"^(\d+)_", p.name)
        d = _run_json(p.name) if (p.is_dir() and mm) else None
        text = (d or {}).get("meta", {}).get("Prompt")
        if text:
            found.append((int(mm.group(1)), text, d.get("time", "")))
    found.sort(key=lambda t: -t[0])
    seen, out = {}, []
    for _, text, when in found:
        if text in seen:
            seen[text]["runs"] += 1
        else:
            seen[text] = {"prompt": text, "last_used": when, "runs": 1}
            out.append(seen[text])
    return {"prompts": out[:30]}


@router.get("/api/runs")
def runs():
    out = []
    for p in core.OUTPUT_DIR.iterdir():
        mm = re.match(r"^(\d+)_", p.name)
        if not (p.is_dir() and mm):
            continue
        d = _run_json(p.name)
        job = JOBS.get(p.name)
        if job and job["state"] == "running":
            out.append((int(mm.group(1)), {"folder": p.name, "state": "running", "title": "Running...",
                                           "meta": {"Prompt": job["prompt"]}, "done": job["done"],
                                           "total": job["total"], "items": []}))
        elif d:
            out.append((int(mm.group(1)), {
                "folder": p.name, "state": "done", "title": d["title"], "meta": d["meta"],
                "total_inr": d["total_inr"], "time": d.get("time", ""), "input": d["input"], "items": d["items"],
            }))
    out.sort(key=lambda t: -t[0])
    return {"runs": [r for _, r in out]}


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
            "input_url": f"/outputs/{folder}/{d['input']}", "branches": branches}


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


def _export_rows(folder_names: list) -> tuple[list, list]:
    """One row per output image (or per failed image). Returns (rows, run_summaries)."""
    rows, runs_ = [], []
    for name in folder_names:
        d = _run_json(name)
        if not d:
            continue
        meta = d.get("meta", {})
        base = {"folder": name, "time": d.get("time", ""), "prompt": meta.get("Prompt", ""),
                "input": d.get("input"), "size": meta.get("Size", "")}
        branches = d.get("branches")
        if not branches:
            branches = [{"kind": "direct", "label": "Direct: your prompt, no text model", "final_prompt": "",
                         "items": d.get("items", [])}]
        n_img = 0
        for b in branches:
            text_cost = b.get("text_cost_inr")
            first = True
            items = b.get("items") or [{"label": "", "file": None, "error": b.get("error")}]
            for it in items:
                n_img += 1 if it.get("file") else 0
                rows.append({**base,
                             "branch": b.get("label", ""),
                             "text_model": b.get("text_model") or "(none, direct)",
                             "effort": b.get("effort") or "",
                             "analysis": b.get("analysis") or "",
                             "final_prompt": b.get("final_prompt") or "",
                             "image_model": it.get("model") or meta.get("Model", ""),
                             "quality": it.get("quality") or meta.get("Quality", ""),
                             "file": it.get("file"),
                             "label": it.get("label", ""),
                             "status": "ok" if it.get("file") else (it.get("error") or b.get("error") or it.get("status") or "not generated"),
                             "text_cost": text_cost if (first and text_cost is not None) else None,
                             "image_cost": it.get("cost_inr")})
                first = False
        runs_.append({"folder": name, "time": d.get("time", ""), "prompt": meta.get("Prompt", ""),
                      "branches": len(branches), "images": n_img, "total": d.get("total_inr", 0)})
    return rows, runs_


# ---------------------------------------------------------------- reviews (feedback sheet)
REVIEW_FILE = Path(core.COST_LOG).parent / "reviews.json"
_REVIEW_LOCK = threading.Lock()
REVIEW_FIELDS = [
    ("satisfied", "Satisfied with the output"),
    ("object_ok", "Correct object identification"),
    ("separation_ok", "Correct image separation"),
    ("background_ok", "Happy with the background"),
    ("product_unchanged", "Product unchanged"),
    ("lighting_natural", "Lighting and shadows look natural"),
    ("usable_as_ad", "Usable as an ad without edits"),
]
_REVIEW_KEYS = [k for k, _ in REVIEW_FIELDS]


def _load_reviews() -> dict:
    try:
        return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _review_key(folder: str, file: str) -> str:
    return f"{folder}|{file}"


def _latest(data: dict, folder: str, file: str) -> dict:
    """The review for one image. One person reviews, but if old multi-name records exist the newest wins."""
    recs = [v for k, v in data.items() if k.startswith(f"{folder}|{file}")]
    return max(recs, key=lambda v: v.get("updated_ts", 0)) if recs else {}


def _is_reviewed(rec: dict) -> bool:
    return bool(rec) and (any(rec.get(k) for k in _REVIEW_KEYS) or bool(rec.get("comment")))


@router.get("/api/review/items")
def review_items(run: str = "", status: str = ""):
    """One entry per generated image with its saved review. status: todo (not reviewed) | done (reviewed)."""
    if run:
        _check_folder(run)
    all_runs = sorted((p.name for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)),
                      key=lambda n: -int(n.split("_", 1)[0]))
    rows, _ = _export_rows([run] if run else all_runs)
    reviews = _load_reviews()
    items, done = [], 0
    for r in rows:
        if not r["file"]:
            continue
        rec = _latest(reviews, r["folder"], r["file"])
        reviewed = _is_reviewed(rec)
        done += reviewed
        if (status == "todo" and reviewed) or (status == "done" and not reviewed):
            continue
        items.append({"folder": r["folder"], "time": r["time"], "prompt": r["prompt"], "size": r["size"],
                      "input_url": f"/outputs/{r['folder']}/{r['input']}" if r["input"] else None,
                      "text_model": r["text_model"], "effort": r["effort"], "analysis": r["analysis"],
                      "final_prompt": r["final_prompt"], "image_model": r["image_model"], "quality": r["quality"],
                      "file": r["file"], "url": f"/outputs/{r['folder']}/{r['file']}",
                      "review": rec if reviewed else {}, "reviewed": reviewed})
    total = sum(1 for r in rows if r["file"])
    return {"fields": [{"key": k, "label": l} for k, l in REVIEW_FIELDS], "items": items, "runs": all_runs,
            "total": total, "reviewed_count": done}


@router.post("/api/review")
async def save_review(payload: dict):
    """Save (replace) the review of one image: {folder, file, answers:{key: yes|no|""}, comment}."""
    folder = str(payload.get("folder", ""))
    file = str(payload.get("file", ""))
    _check_folder(folder)
    if not re.match(r"^[\w.\-+]+\.(png|jpg|jpeg|webp)$", file, re.I) or not (core.OUTPUT_DIR / folder / file).is_file():
        raise HTTPException(404, "Image not found.")
    answers = payload.get("answers") or {}
    rec = {k: (answers.get(k) if answers.get(k) in ("yes", "no") else "") for k in _REVIEW_KEYS}
    rec["comment"] = str(payload.get("comment", "")).strip()[:2000]
    if not _is_reviewed(rec):
        raise HTTPException(400, "Answer at least one question (or write a comment) before saving.")
    now = datetime.now()
    rec.update({"folder": folder, "file": file, "updated": now.strftime("%d %b %Y, %H:%M"), "updated_ts": now.timestamp()})
    with _REVIEW_LOCK:
        data = _load_reviews()
        for k in [k for k in data if k.startswith(f"{folder}|{file}")]:      # replace any older record
            del data[k]
        data[_review_key(folder, file)] = rec
        tmp = REVIEW_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(REVIEW_FILE)
    return {"ok": True, "saved": rec}


@router.get("/review.xlsx")
def review_xlsx(request: Request, run: str = "", pictures: int = 1, reviewed_only: int = 0):
    """Excel: one row per image with its review (images not reviewed yet have blank answers unless reviewed_only=1)."""
    import xlsxwriter
    if run:
        _check_folder(run)
    names = [run] if run else sorted((p.name for p in core.OUTPUT_DIR.iterdir()
                                      if p.is_dir() and re.match(r"^\d+_", p.name)),
                                     key=lambda n: int(n.split("_", 1)[0]))
    rows, _ = _export_rows(names)
    reviews = _load_reviews()
    base = _public_base(request)
    out = []                                   # (image row, review dict or {})
    for r in rows:
        if not r["file"]:
            continue
        rec = _latest(reviews, r["folder"], r["file"])
        if _is_reviewed(rec):
            out.append((r, rec))
        elif not reviewed_only:
            out.append((r, {}))

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#1f2a44", "font_color": "#ffffff", "valign": "vcenter", "text_wrap": True})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    top = wb.add_format({"valign": "top"})
    mid = wb.add_format({"valign": "top", "align": "center"})
    link = wb.add_format({"font_color": "#0563c1", "underline": 1, "valign": "top"})
    green = wb.add_format({"valign": "top", "align": "center", "bg_color": "#d9f2e3", "bold": True})
    red = wb.add_format({"valign": "top", "align": "center", "bg_color": "#f9d9d6", "bold": True})

    def cap(t):
        t = str(t or "")
        return t if len(t) <= 32000 else t[:32000] + " ...[cut]"

    ws = wb.add_worksheet("Review")
    cols = [("Run id", 22), ("Date", 17), ("Input image", 24), ("Initial prompt", 40), ("Text model", 14), ("Effort", 8),
            ("Analysis (by text model)", 60), ("Prompt given to the image model", 60), ("Image model", 22),
            ("Quality", 9), ("Size", 11), ("Output image", 24), ("Output link", 14)]
    cols += [(l, 15) for _, l in REVIEW_FIELDS] + [("Comment", 40), ("Reviewed at", 17)]
    for i, (t, w) in enumerate(cols):
        ws.write(0, i, t, hdr)
        ws.set_column(i, i, w)
    ws.set_row(0, 46)
    ws.freeze_panes(1, 1)
    ROW_PX = 190

    def put_image(r, c, path):
        sz = _png_size(path)
        if not sz:
            return False
        w, h = sz
        sc = min(ROW_PX / h, 170 / w)
        ws.insert_image(r, c, str(path), {"x_scale": sc, "y_scale": sc, "x_offset": 3, "y_offset": 3, "object_position": 1})
        return True

    for n, (r, rv) in enumerate(out, start=1):
        ws.set_row(n, ROW_PX * 0.75 + 6 if pictures else 90)
        ws.write(n, 0, r["folder"], top)
        ws.write(n, 1, r["time"], top)
        in_path = core.OUTPUT_DIR / r["folder"] / (r["input"] or "")
        if pictures and r["input"] and in_path.is_file():
            if not put_image(n, 2, in_path):
                ws.write(n, 2, r["input"], top)
        elif r["input"] and base:
            ws.write_url(n, 2, f"{base}/outputs/{r['folder']}/{r['input']}", link, "open input")
        ws.write(n, 3, cap(r["prompt"]), wrap)
        ws.write(n, 4, r["text_model"], top)
        ws.write(n, 5, r["effort"], top)
        ws.write(n, 6, cap(r["analysis"]), wrap)
        ws.write(n, 7, cap(r["final_prompt"]), wrap)
        ws.write(n, 8, r["image_model"], top)
        ws.write(n, 9, r["quality"], top)
        ws.write(n, 10, r["size"], top)
        out_path = core.OUTPUT_DIR / r["folder"] / r["file"]
        if out_path.is_file():
            if pictures:
                put_image(n, 11, out_path)
            if base:
                ws.write_url(n, 12, f"{base}/outputs/{r['folder']}/{r['file']}", link, "open image")
        for j, (k, _) in enumerate(REVIEW_FIELDS):
            v = rv.get(k, "")
            ws.write(n, 13 + j, {"yes": "Yes", "no": "No"}.get(v, ""), green if v == "yes" else red if v == "no" else mid)
        ws.write(n, 13 + len(REVIEW_FIELDS), cap(rv.get("comment", "")), wrap)
        ws.write(n, 14 + len(REVIEW_FIELDS), rv.get("updated", ""), top)
    if out:
        ws.autofilter(0, 0, len(out), len(cols) - 1)

    sm = wb.add_worksheet("Summary")                    # Yes / No counts per question
    for i, t in enumerate(["Question", "Yes", "No", "Not answered", "% Yes (of answered)"]):
        sm.write(0, i, t, hdr)
    sm.set_column(0, 0, 40)
    sm.set_column(1, 4, 16)
    pct = wb.add_format({"num_format": "0%"})
    last = len(out) + 1
    for j, (k, label) in enumerate(REVIEW_FIELDS):
        col = chr(ord("N") + j)                         # N.. = first answer column
        r_ = j + 1
        sm.write(r_, 0, label)
        sm.write_formula(r_, 1, f'=COUNTIF(Review!{col}2:{col}{last},"Yes")')
        sm.write_formula(r_, 2, f'=COUNTIF(Review!{col}2:{col}{last},"No")')
        sm.write_formula(r_, 3, f'={max(len(out), 0)}-B{r_ + 1}-C{r_ + 1}')
        sm.write_formula(r_, 4, f'=IF(B{r_ + 1}+C{r_ + 1}=0,"",B{r_ + 1}/(B{r_ + 1}+C{r_ + 1}))', pct)
    wb.close()
    fname = f"review_{run or 'all_runs'}.xlsx"
    return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.get("/api/costlog")
def costlog():
    rows = _cost_rows()
    def _sum(kind):
        return round(sum(float(r.get("cost_inr") or 0) for r in rows if (r.get("step") or "image") == kind), 2)
    return {"total_inr": round(total_so_far(core.COST_LOG), 2), "analysis_inr": _sum("analysis"),
            "images_inr": _sum("image"), "rows": rows[-500:][::-1], "count": len(rows)}


@router.get("/cost.csv")
def cost_csv():
    p = Path(core.COST_LOG)
    data = p.read_bytes() if p.is_file() else (",".join(LOG_FIELDS) + "\n").encode()
    return Response(data, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="cost_log.csv"'})
