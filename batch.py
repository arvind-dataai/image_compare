"""Batch mode: ONE prompt x MANY images, one model (gpt-image-2.5-sunburst, low quality), plus its own review sheet.

  POST /api/batch                       prompt + many images -> starts a background batch, returns {folder}
  GET  /api/batch                       list of batches
  GET  /api/batch/{folder}              live status of one batch (poll while running)
  POST /api/batch/{folder}/retry        re-run only the failed images of that batch
  GET  /api/batchreview/items           one entry per generated image with its saved review
  POST /api/batchreview                 save / replace one review
  GET  /batch_review.xlsx               Excel of this batch review sheet

Kept completely apart from the normal runs and the normal Review tab:
  * folders are  outputs/batch_<n>_<name>/  (the normal pages only list folders named  <n>_...)
  * reviews are saved in batch_reviews.json (the normal ones stay in reviews.json)

The prompt is sent exactly as typed: no text model, no extra rules added.
"""
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

import main as core

router = APIRouter()

BATCH_MODEL = "gpt-image-2.5-sunburst"
BATCH_QUALITY = "low"
BATCH_MAX = int(os.getenv("WEB_BATCH_MAX", "100"))                       # safety cap on images in one batch
BATCH_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WEB_BATCH_WORKERS", "3")))
_BFOLDER = re.compile(r"^batch_\d+_[A-Za-z0-9_.+\-]+$")
_LOCK = threading.Lock()                                                 # guards batch.json writes + folder numbering
BJOBS: dict = {}                                                         # folder -> live state (also saved in batch.json)

REVIEW_FILE = Path(core.COST_LOG).parent / "batch_reviews.json"
_REVIEW_LOCK = threading.Lock()
REVIEW_FIELDS = [                                   # same questions as the normal Review tab
    ("satisfied", "Satisfied with the output"),
    ("object_ok", "Correct object identification"),
    ("separation_ok", "Correct image separation"),
    ("background_ok", "Happy with the background"),
    ("product_unchanged", "Product unchanged"),
    ("lighting_natural", "Lighting and shadows look natural"),
    ("usable_as_ad", "Usable as an ad without edits"),
]
_KEYS = [k for k, _ in REVIEW_FIELDS]


# ---------------------------------------------------------------- small helpers
def _dir(folder: str) -> Path:
    if not _BFOLDER.match(folder) or not (core.OUTPUT_DIR / folder).is_dir():
        raise HTTPException(404, "Batch not found.")
    return core.OUTPUT_DIR / folder


def _read(folder: str) -> dict | None:
    try:
        return json.loads((core.OUTPUT_DIR / folder / "batch.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def _write(folder: str, data: dict):
    p = core.OUTPUT_DIR / folder / "batch.json"
    tmp = p.with_suffix(".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def _all_folders() -> list:
    names = [p.name for p in core.OUTPUT_DIR.iterdir() if p.is_dir() and _BFOLDER.match(p.name)]
    return sorted(names, key=lambda n: -int(n.split("_")[1]))


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:24] or "batch")


def _new_folder(name: str) -> Path:
    with _LOCK:
        nums = [int(n.split("_")[1]) for n in (p.name for p in core.OUTPUT_DIR.iterdir() if _BFOLDER.match(p.name))]
        d = core.OUTPUT_DIR / f"batch_{max(nums, default=0) + 1}_{_slug(name)}"
        d.mkdir()
    return d


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", Path(name or "image").stem)[:40] or "image"


def _view(job: dict) -> dict:
    """What the browser needs: the job without file paths it cannot use."""
    f = job["folder"]
    out = dict(job)
    out["items"] = [dict(it, input_url=f"/outputs/{f}/{it['input']}",
                         url=(f"/outputs/{f}/{it['file']}" if it.get("file") else None)) for it in job["items"]]
    return out


# ---------------------------------------------------------------- the background job
def _count(job: dict):
    job["done"] = sum(1 for it in job["items"] if it["status"] in ("done", "error"))


def _run_items(folder: str, indexes: list):
    job = BJOBS[folder]
    run_dir = core.OUTPUT_DIR / folder
    prompt, size = job["prompt"], job["size"]

    def one(i):
        it = job["items"][i]
        it["status"], it["error"] = "running", ""
        try:
            data = (run_dir / it["input"]).read_bytes()
            ctype = "image/png" if it["input"].lower().endswith(".png") else \
                    "image/webp" if it["input"].lower().endswith(".webp") else "image/jpeg"
            png, usage = core._generate((it["input"], data, ctype), prompt, size, BATCH_MODEL, BATCH_QUALITY)
            fname = f"out_{i + 1:03d}_{_safe_name(it['input'])}.png"
            cost = core._store(run_dir, fname, png, usage, size, BATCH_MODEL, BATCH_QUALITY, f"batch | {folder}")
            it.update(status="done", file=fname, cost_inr=round(cost, 4))
        except Exception as e:                       # one failed image must not stop the others
            it.update(status="error", error=str(getattr(e, "detail", None) or e)[:500])
        _count(job)
        job["total_inr"] = round(sum(x.get("cost_inr") or 0 for x in job["items"]), 4)
        _write(folder, job)

    try:
        list(BATCH_POOL.map(one, indexes))
        job["state"] = "done"
    except Exception as e:                           # should not happen, but never leave a batch "running" forever
        job["state"] = "error"
        job["error"] = str(e)
    _count(job)
    _write(folder, job)


def _start(folder: str, indexes: list):
    BJOBS[folder]["state"] = "running"
    threading.Thread(target=_run_items, args=(folder, indexes), daemon=True).start()


def _load(folder: str) -> dict:
    """Live job if this server process started it, else the saved batch.json (a 'running' one left over
    from before a restart is shown as interrupted so it never looks stuck)."""
    if folder in BJOBS:
        return BJOBS[folder]
    data = _read(folder)
    if not data:
        raise HTTPException(404, "Batch not found.")
    if data.get("state") == "running":
        data["state"] = "interrupted"
        for it in data["items"]:
            if it["status"] in ("queued", "running"):
                it["status"], it["error"] = "error", "Interrupted: the server restarted before this image finished."
        _count(data)
    return data


# ---------------------------------------------------------------- API: run
@router.get("/api/batch/options")
def batch_options():
    from webapp import SIZES
    from costing import estimate_inr
    return {"model": BATCH_MODEL, "quality": BATCH_QUALITY, "max_images": BATCH_MAX,
            "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
            "est_inr_1024": round(estimate_inr(BATCH_QUALITY, BATCH_MODEL), 4)}


@router.post("/api/batch")
async def batch_start(
    images: list[UploadFile] = File(...),
    prompt: str = Form(...),
    size: str = Form("1088x1360"),
    name: str = Form(""),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Enter the prompt.")
    if not images:
        raise HTTPException(400, "Add at least one image.")
    if len(images) > BATCH_MAX:
        raise HTTPException(400, f"Too many images ({len(images)}). The limit is {BATCH_MAX} per batch.")
    core._check_size(size)
    read = []
    for up in images:                                # validates type + size like the normal form
        read.append(await core._read_image(up))
    d = _new_folder(name or prompt[:24])
    folder = d.name
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{_safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "status": "queued", "error": "", "file": None, "cost_inr": 0})
    job = {"folder": folder, "state": "running", "name": name.strip() or prompt[:40], "prompt": prompt, "size": size,
           "model": BATCH_MODEL, "quality": BATCH_QUALITY, "created": datetime.now().strftime("%d %b %Y, %H:%M"),
           "total": len(items), "done": 0, "total_inr": 0, "items": items}
    BJOBS[folder] = job
    _write(folder, job)
    _start(folder, list(range(len(items))))
    return {"folder": folder, "images": len(items)}


@router.get("/api/batch")
def batch_list():
    out = []
    for f in _all_folders():
        try:
            j = _load(f)
        except HTTPException:
            continue
        out.append({"folder": f, "state": j["state"], "name": j.get("name", ""), "created": j.get("created", ""),
                    "prompt": j["prompt"], "total": j["total"], "done": j.get("done", 0),
                    "failed": sum(1 for it in j["items"] if it["status"] == "error"),
                    "total_inr": j.get("total_inr", 0)})
    return {"batches": out}


@router.get("/api/batch/{folder}")
def batch_status(folder: str):
    _dir(folder)
    return _view(_load(folder))


@router.post("/api/batch/{folder}/retry")
def batch_retry(folder: str):
    _dir(folder)
    job = _load(folder)
    if folder in BJOBS and BJOBS[folder]["state"] == "running":
        raise HTTPException(400, "This batch is still running.")
    BJOBS[folder] = job                              # adopt a saved (finished / interrupted) batch
    bad = [i for i, it in enumerate(job["items"]) if it["status"] != "done"]
    if not bad:
        raise HTTPException(400, "Nothing to retry: every image is done.")
    for i in bad:
        job["items"][i].update(status="queued", error="")
    job.pop("error", None)
    _count(job)
    _start(folder, bad)
    return {"folder": folder, "retrying": len(bad)}


# ---------------------------------------------------------------- API: review sheet
def _load_reviews() -> dict:
    try:
        return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_reviewed(rec: dict) -> bool:
    return bool(rec) and (any(rec.get(k) for k in _KEYS) or bool(rec.get("comment")))


def _rows(folders: list) -> list:
    out = []
    for f in folders:
        j = _read(f)
        if not j:
            continue
        for it in j["items"]:
            if it.get("status") == "done" and it.get("file"):
                out.append({"folder": f, "name": j.get("name", ""), "time": j.get("created", ""), "prompt": j["prompt"],
                            "size": j["size"], "model": j.get("model", BATCH_MODEL), "quality": j.get("quality", BATCH_QUALITY),
                            "input": it["input"], "orig_name": it.get("orig_name", ""), "file": it["file"],
                            "cost_inr": it.get("cost_inr", 0)})
    return out


@router.get("/api/batchreview/items")
def batchreview_items(batch: str = "", status: str = ""):
    if batch:
        _dir(batch)
    folders = _all_folders()
    rows = _rows([batch] if batch else folders)
    reviews = _load_reviews()
    items, done = [], 0
    for r in rows:
        rec = reviews.get(f"{r['folder']}|{r['file']}", {})
        rv = _is_reviewed(rec)
        done += rv
        if (status == "todo" and rv) or (status == "done" and not rv):
            continue
        items.append({**r, "input_url": f"/outputs/{r['folder']}/{r['input']}", "url": f"/outputs/{r['folder']}/{r['file']}",
                      "review": rec if rv else {}, "reviewed": rv})
    return {"fields": [{"key": k, "label": l} for k, l in REVIEW_FIELDS], "items": items, "batches": folders,
            "total": len(rows), "reviewed_count": done}


@router.post("/api/batchreview")
async def batchreview_save(payload: dict):
    folder, file = str(payload.get("folder", "")), str(payload.get("file", ""))
    _dir(folder)
    if not re.match(r"^[\w.\-+]+\.png$", file, re.I) or not (core.OUTPUT_DIR / folder / file).is_file():
        raise HTTPException(404, "Image not found.")
    answers = payload.get("answers") or {}
    rec = {k: (answers.get(k) if answers.get(k) in ("yes", "no") else "") for k in _KEYS}
    rec["comment"] = str(payload.get("comment", "")).strip()[:2000]
    if not _is_reviewed(rec):
        raise HTTPException(400, "Answer at least one question (or write a comment) before saving.")
    now = datetime.now()
    rec.update({"folder": folder, "file": file, "updated": now.strftime("%d %b %Y, %H:%M"), "updated_ts": now.timestamp()})
    with _REVIEW_LOCK:
        data = _load_reviews()
        data[f"{folder}|{file}"] = rec
        tmp = REVIEW_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(REVIEW_FILE)
    return {"ok": True, "saved": rec}


@router.get("/batch_review.xlsx")
def batchreview_xlsx(request: Request, batch: str = "", pictures: int = 1, reviewed_only: int = 0):
    import xlsxwriter
    from webapp import _png_size, _public_base
    if batch:
        _dir(batch)
    rows = _rows([batch] if batch else sorted(_all_folders(), key=lambda n: int(n.split("_")[1])))
    reviews = _load_reviews()
    base = _public_base(request)
    out = []
    for r in rows:
        rec = reviews.get(f"{r['folder']}|{r['file']}", {})
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
    cap = lambda t: (lambda s: s if len(s) <= 32000 else s[:32000] + " ...[cut]")(str(t or ""))

    ws = wb.add_worksheet("Batch review")
    cols = [("Batch id", 24), ("Date", 17), ("Input image", 24), ("Input file", 22), ("Prompt", 60), ("Image model", 22),
            ("Quality", 9), ("Size", 11), ("Output image", 24), ("Output link", 14), ("Cost (INR)", 10)]
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
        in_path = core.OUTPUT_DIR / r["folder"] / r["input"]
        if pictures and in_path.is_file():
            if not put_image(n, 2, in_path):
                ws.write(n, 2, r["input"], top)
        elif base:
            ws.write_url(n, 2, f"{base}/outputs/{r['folder']}/{r['input']}", link, "open input")
        ws.write(n, 3, r["orig_name"], top)
        ws.write(n, 4, cap(r["prompt"]), wrap)
        ws.write(n, 5, r["model"], top)
        ws.write(n, 6, r["quality"], top)
        ws.write(n, 7, r["size"], top)
        out_path = core.OUTPUT_DIR / r["folder"] / r["file"]
        if out_path.is_file():
            if pictures:
                put_image(n, 8, out_path)
            if base:
                ws.write_url(n, 9, f"{base}/outputs/{r['folder']}/{r['file']}", link, "open image")
        ws.write(n, 10, r["cost_inr"], top)
        for j, (k, _) in enumerate(REVIEW_FIELDS):
            v = rv.get(k, "")
            ws.write(n, 11 + j, {"yes": "Yes", "no": "No"}.get(v, ""), green if v == "yes" else red if v == "no" else mid)
        ws.write(n, 11 + len(REVIEW_FIELDS), cap(rv.get("comment", "")), wrap)
        ws.write(n, 12 + len(REVIEW_FIELDS), rv.get("updated", ""), top)
    if out:
        ws.autofilter(0, 0, len(out), len(cols) - 1)

    sm = wb.add_worksheet("Summary")
    for i, t in enumerate(["Question", "Yes", "No", "Not answered", "% Yes (of answered)"]):
        sm.write(0, i, t, hdr)
    sm.set_column(0, 0, 40)
    sm.set_column(1, 4, 16)
    pct = wb.add_format({"num_format": "0%"})
    last = len(out) + 1
    for j, (k, label) in enumerate(REVIEW_FIELDS):
        col = chr(ord("L") + j)                      # L.. = first answer column
        r_ = j + 1
        sm.write(r_, 0, label)
        sm.write_formula(r_, 1, f"=COUNTIF('Batch review'!{col}2:{col}{last},\"Yes\")")
        sm.write_formula(r_, 2, f"=COUNTIF('Batch review'!{col}2:{col}{last},\"No\")")
        sm.write_formula(r_, 3, f"={len(out)}-B{r_ + 1}-C{r_ + 1}")
        sm.write_formula(r_, 4, f'=IF(B{r_ + 1}+C{r_ + 1}=0,"",B{r_ + 1}/(B{r_ + 1}+C{r_ + 1}))', pct)
    wb.close()
    fname = f"batch_review_{batch or 'all_batches'}.xlsx"
    return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})
