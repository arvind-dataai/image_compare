"""Small web UI + API on top of the existing backend.

  GET  /                      the page (web/index.html)
  GET  /api/options           models, qualities, rough per-image price estimates
  POST /api/compare           image + prompt + models + qualities -> starts a run, returns {folder}
  GET  /api/runs              history of all runs
  GET  /api/runs/{folder}     live status of one run (poll this while it is generating)
  GET  /api/runs/{folder}/zip everything in that run's folder
  GET  /api/costlog           cost_log.csv as JSON rows (for the Cost log tab)
  GET  /cost.csv              download cost_log.csv

One run = every selected model x every selected quality, all in one numbered folder, with a
shareable comparison.html (input + all images + cost table), exactly like the other endpoints.

Optional password: set APP_PASSWORD in .env and the browser asks for it (user name can be anything).
"""
import base64
import csv
import io
import json
import os
import re
import secrets
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

import main as core
from costing import ESTIMATE_1024, LOG_FIELDS, USD_TO_INR, total_so_far
from prompts import build_custom_prompt
from runs import new_run_dir, short_model, write_comparison

router = APIRouter()

# What the page offers. To add a model or quality, add it here (and to the enums in main.py).
WEB_MODELS = {
    "gpt-image-2.5-flare": "Flare: fast, good for everyday edits",
    "gpt-image-2.5-sunburst": "Sunburst: most capable, precise edits",
}
WEB_QUALITIES = ["low", "medium", "high"]
SIZES = {
    "1024x1024": "Square 1:1 (1024x1024)",
    "1088x1360": "Feed 4:5 (1088x1360)",
    "1024x1536": "Portrait 2:3 (1024x1536)",
    "1536x1024": "Landscape 3:2 (1536x1024)",
}

POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WEB_WORKERS", "4")))
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
def _run_job(folder: str, run_dir: Path, img, in_name: str, prompt: str, size: str, combos: list):
    job = JOBS[folder]
    full_prompt = build_custom_prompt(prompt)
    ledger: list = []

    def one(idx_combo):
        idx, (m, q) = idx_combo
        item = job["items"][idx]
        item["status"] = "running"
        try:
            png, usage = core._generate(img, full_prompt, size, m, q)
            fname = f"{short_model(m)}_{q}.png"
            cost = core._store(run_dir, fname, png, usage, size, m, q, "web", ledger)
            item.update(status="done", file=fname, url=f"/outputs/{folder}/{fname}", cost_inr=round(cost, 4))
        except Exception as e:                       # one failed image must not stop the others
            detail = getattr(e, "detail", None) or str(e)
            item.update(status="error", error=str(detail))
        job["done"] += 1

    try:
        list(POOL.map(one, enumerate(combos)))
        items = [{"label": it["label"], "file": it.get("file"), "cost_inr": it.get("cost_inr"),
                  "error": it.get("error"), "prompt": ""} for it in job["items"]]
        ledger.sort(key=lambda c: c["part"])
        total = sum(c["cost_inr"] for c in ledger)
        write_comparison(
            run_dir, f"Compare: {prompt[:70]}", in_name, items,
            {"Prompt": prompt, "Size": size, "Images": len(combos), "Total cost": f"₹{total:.2f}"},
            costs=ledger, prompt=prompt, full_prompt=full_prompt,
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
        "models": [{"id": k, "label": v} for k, v in WEB_MODELS.items()],
        "qualities": WEB_QUALITIES,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
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
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Enter a prompt.")
    models = [m for m in WEB_MODELS if m in models]            # keeps a fixed order, drops unknowns
    qualities = [q for q in WEB_QUALITIES if q in qualities]
    if not models or not qualities:
        raise HTTPException(400, "Select at least one model and one quality.")
    core._check_size(size)
    img = await core._read_image(image)

    combos = [(m, q) for m in models for q in qualities]
    mpart = "gpt2.5" + "+".join(short_model(m).replace("gpt2.5", "") for m in models)
    run_dir = new_run_dir(core.OUTPUT_DIR, mpart, "+".join(qualities))
    folder = run_dir.name
    in_name = core._save_input(run_dir, img)
    JOBS[folder] = {
        "folder": folder, "state": "running", "total": len(combos), "done": 0, "prompt": prompt, "size": size,
        "input_url": f"/outputs/{folder}/{in_name}",
        "items": [{"label": f"{short_model(m)} · {q}", "model": m, "quality": q, "status": "queued"}
                  for m, q in combos],
    }
    threading.Thread(target=_run_job, args=(folder, run_dir, img, in_name, prompt, size, combos),
                     daemon=True).start()
    return {"folder": folder, "images": len(combos)}


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
    return {"folder": folder, "state": "done", "total_inr": d["total_inr"], "prompt": d["meta"].get("Prompt", ""),
            "input_url": f"/outputs/{folder}/{d['input']}",
            "items": [{**it, "status": "done" if it.get("file") else "error",
                       "url": f"/outputs/{folder}/{it['file']}" if it.get("file") else None} for it in d["items"]]}


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


@router.get("/api/costlog")
def costlog():
    rows = _cost_rows()
    return {"total_inr": round(total_so_far(core.COST_LOG), 2), "rows": rows[-500:][::-1], "count": len(rows)}


@router.get("/cost.csv")
def cost_csv():
    p = Path(core.COST_LOG)
    data = p.read_bytes() if p.is_file() else (",".join(LOG_FIELDS) + "\n").encode()
    return Response(data, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="cost_log.csv"'})
