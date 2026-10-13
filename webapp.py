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
from costing import (ESTIMATE_1024, LOG_FIELDS, TEXT_RATES_PER_M, USD_TO_INR, _as_dict, log_text_cost,
                     total_so_far)
from prompts import build_custom_prompt
from runs import new_run_dir, short_model, write_flow_comparison

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
        "models": [{"id": k, "label": v} for k, v in WEB_MODELS.items()],
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

    mpart = "gpt2.5" + "+".join(short_model(m).replace("gpt2.5", "") for m in models)
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
