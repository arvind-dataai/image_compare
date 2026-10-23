"""Batch "scene flow": every uploaded photo is its OWN product -> identify -> new scene -> image, one product at a time.

  GET  /api/scene/options     text models, default prompts, sizes
  POST /api/scene/run         photos (one product each) + the two text prompts + Prompt B -> starts the flow, returns {folder}
  GET  /api/scene/library     category > subcategory > scenes (for the Scene library tab)

Per photo, three steps (each one is saved as it finishes, so "Retry" continues from the step that failed):
  1 classify : text model (prompt A1 + known taxonomy) -> category, subcategory, product description
  2 scene    : text model (prompt A2 + earlier scenes of that subcategory + Prompt B) -> structured new scene
  3 image    : Prompt C = rendered scene + "Additional instructions:" + Prompt B -> gpt-image-2.5-sunburst (low)

Order inside one run: all photos are identified first; then photos that landed in the same subcategory are done one
after the other (so each new scene sees the scenes just written for the earlier photos), while different subcategories run in parallel.

The run is stored like any batch (outputs/batch_<n>_<name>/batch.json, flow = "scene"), so the Batch list, the
Batch review tab and its Excel already handle it. The scenes themselves live in scene_library.json (scene_library.py).
"""
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import batch
import main as core
import scene_library as lib
from costing import log_text_cost, text_rate

router = APIRouter()

HERE = Path(__file__).parent
DEFAULT_A1 = (HERE / "scene_classify_prompt.txt").read_text(encoding="utf-8")
DEFAULT_A2 = (HERE / "scene_write_prompt.txt").read_text(encoding="utf-8")
PREV_LIMIT = 40
EST_IN = {"classify": 2500, "scene": 4500}                              # rough tokens for the price preview only
EST_OUT = {"none": 500, "minimal": 600, "low": 900, "medium": 1800, "high": 3500, "xhigh": 7000, "max": 12000}

CLASSIFY_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after):\n'
    '{"category": "...", "subcategory": "...", "product_description": "one or two sentences: what the product is, its colours, '
    'materials and number of pieces, as visible", "reason": "one sentence on why it belongs there"}'
)
SCENE_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after). Every value is a string with real detail '
    '(one to three sentences), except where a list is shown:\n'
    '{\n'
    '  "concept_name": "a short name for this scene",\n'
    '  "setting_environment": "the place and surroundings of the scene",\n'
    '  "surface_and_background": "what the product rests on and what is behind it, with materials and colours",\n'
    '  "product_presentation": "how the product is arranged and supported so it looks correct and professional",\n'
    '  "composition": "where the product sits in the 4:5 frame, how large it is, the balance of the picture",\n'
    '  "camera_angle_and_framing": "viewpoint, height, distance and lens feel",\n'
    '  "lighting": "direction, quality and colour of the light, and the shadows",\n'
    '  "colour_palette": "the main colours of the scene and why they contrast with the product",\n'
    '  "props": "none, or the few plain props and exactly where they are",\n'
    '  "mood_and_style": "the feeling and photographic style",\n'
    '  "keep_identical": "the product details that must stay exactly as in the photos",\n'
    '  "avoid": "what must not appear or change",\n'
    '  "difference_from_previous": "how this scene differs from the previous scenes (name them)"\n'
    '}'
)


def _models():
    from webapp import TEXT_MODELS, TEXT_PROVIDER, _text_ready
    return TEXT_MODELS, TEXT_PROVIDER, _text_ready


@router.get("/api/scene/options")
def scene_options():
    from webapp import EFFORTS, SIZES
    tm, prov, ready = _models()
    return {
        "text_models": [{"id": k, "label": v["label"], "efforts": v["efforts"], "available": ready(k),
                         "needs": ("GEMINI_API_KEY" if prov[k] == "google" else "MODEL_API_KEY" if prov[k] == "meta" else "") + " in the server's .env",
                         "rate_in": text_rate(k)[0], "rate_out": text_rate(k)[1]} for k, v in tm.items()],
        "efforts": EFFORTS, "est_in": EST_IN, "est_out": EST_OUT, "usd_to_inr": core.USD_TO_INR,
        "sizes": [{"id": k, "label": v} for k, v in SIZES.items()],
        "prompt_a1": DEFAULT_A1, "prompt_a2": DEFAULT_A2, "max_photos": batch.BATCH_MAX,
        "image_model": batch.BATCH_MODEL, "quality": batch.BATCH_QUALITY,
        "est_image_inr_1024": batch_est(),
    }


def batch_est():
    from costing import estimate_inr
    return round(estimate_inr(batch.BATCH_QUALITY, batch.BATCH_MODEL), 4)


@router.get("/api/scene/library")
def scene_library_view():
    return {"categories": lib.overview()}


# ---------------------------------------------------------------- the run
def _set(d: dict, **kw):
    with batch._LOCK:                                   # never change a dict while batch.json is being written
        d.update(kw)


def _text_call(job: dict, sc: dict, label: str, effort: str, instructions: str, user_text: str, image, cost_key: str):
    """One text call + its cost. Retries once if the reply is not usable JSON. Returns the parsed dict."""
    cfg = job["scene_cfg"]
    provider = _models()[1][cfg["text_model"]]
    last_err = ""
    for attempt in (1, 2):
        extra = "" if attempt == 1 else (f"\n\nYour previous reply could not be used ({last_err}). "
                                         "Reply again with ONLY the JSON object described above.")
        text, usage = core.generate_text(provider, cfg["text_model"], effort, instructions, user_text + extra, image)
        cost, _, _ = log_text_cost(core.COST_LOG, cfg["text_model"], usage,
                                   note=f"scene flow {label}, effort {effort} | {job['folder']}/{sc.get('input')}")
        with batch._LOCK:
            sc[cost_key] = round(sc.get(cost_key, 0) + cost, 4)
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
        try:
            return lib.parse_json(text)
        except Exception as e:
            last_err = str(e)[:120]
            _set(sc, last_reply=(text or "")[:4000])
    raise RuntimeError(f"The text model did not return valid JSON for the {label} step ({last_err}).")


def _need(d: dict, keys: list, what: str):
    missing = [k for k in keys if not str(d.get(k, "")).strip()]
    if missing:
        raise RuntimeError(f"The {what} reply is missing: {', '.join(missing)}.")


def _image_of(folder: str, it: dict):
    name = it["input"]
    ctype = "image/png" if name.lower().endswith(".png") else "image/webp" if name.lower().endswith(".webp") else "image/jpeg"
    return (name, (core.OUTPUT_DIR / folder / name).read_bytes(), ctype)


def _fail(folder: str, job: dict, it: dict, e: Exception):
    sc = it["scene"]
    detail = str(getattr(e, "detail", None) or e)[:600]
    _set(it, status="error", error=detail)
    if sc.get("scene_id"):                              # the scene was written but its image failed
        lib.update_scene(sc["scene_id"], status="failed")
    batch._count(job)
    batch._write(folder, job)


def _classify(folder: str, job: dict, i: int):
    """Step 1 for item i: ask the text model; keeps the model's raw names for the sequential matching below."""
    it, cfg = job["items"][i], job["scene_cfg"]
    sc = it["scene"]
    try:
        _set(it, status="running", error=""); _set(sc, stage="classify")
        batch._write(folder, job)
        instr = (cfg["prompt_a1"].strip() + "\n\nKNOWN TAXONOMY (categories and their subcategories already in the library):\n"
                 + lib.taxonomy_text() + "\n\n" + CLASSIFY_FORMAT)
        r = _text_call(job, sc, "identify", cfg["effort_classify"], instr,
                       "Identify the single product shown in the attached photo and file it, as instructed.",
                       _image_of(folder, it), "classify_cost_inr")
        _need(r, lib.CLASSIFY_KEYS, "identification")
        _set(sc, raw_category=str(r["category"]), raw_subcategory=str(r["subcategory"]),
             product_description=str(r["product_description"]).strip(), classify_reason=str(r.get("reason", "")).strip())
        batch._write(folder, job)
    except Exception as e:
        _fail(folder, job, it, e)


def _scene_and_image(folder: str, job: dict, i: int):
    """Steps 2 and 3 for item i (its category / subcategory are already settled)."""
    it, cfg = job["items"][i], job["scene_cfg"]
    sc = it["scene"]
    run_dir = core.OUTPUT_DIR / folder
    try:
        image = _image_of(folder, it)
        if "scene_description" not in sc:
            _set(sc, stage="scene"); batch._write(folder, job)
            prev = lib.previous(sc["category"], sc["subcategory"], PREV_LIMIT)
            prev_txt = ("\n".join(f"{n}. {p.get('summary') or lib.summary_of(p.get('scene', {}))}" for n, p in enumerate(prev, 1))
                        if prev else "None yet: this is the first scene for this subcategory.")
            ctx = (f"CONTEXT\nPRODUCT\ncategory: {sc['category']}\nsubcategory: {sc['subcategory']}\ndescription: {sc['product_description']}\n\n"
                   f"USER'S ADDITIONAL INSTRUCTIONS (the system adds these after your scene description):\n{cfg['prompt_b'] or '(none)'}\n\n"
                   f"PREVIOUS SCENES ALREADY USED FOR THIS SUBCATEGORY ({len(prev)}):\n{prev_txt}")
            instr = cfg["prompt_a2"].strip() + "\n\n" + ctx + "\n\n" + SCENE_FORMAT
            r = _text_call(job, sc, "scene", cfg["effort_scene"], instr,
                           "Design the new scene for the product in the attached photo, as instructed.", image, "scene_cost_inr")
            _need(r, lib.SCENE_KEYS, "scene")
            scene = {k: r[k] for k in lib.SCENE_KEYS}
            scene["difference_from_previous"] = str(r.get("difference_from_previous", "")).strip()
            desc = lib.render_scene(sc["category"], sc["subcategory"], sc["product_description"], scene)
            sid = lib.add_scene(sc["category"], sc["subcategory"], {
                "status": "pending", "folder": folder, "input": it["input"], "text_model": cfg["text_model"],
                "effort": cfg["effort_scene"], "prompt_b": cfg["prompt_b"], "scene": scene, "scene_description": desc,
                "summary": lib.summary_of(scene)})
            _set(sc, previous_count=len(prev), scene=scene, scene_description=desc,
                 prompt_c=lib.build_prompt_c(desc, cfg["prompt_b"]), scene_id=sid)
            batch._write(folder, job)
        _set(sc, stage="image"); batch._write(folder, job)
        png, usage = core._generate(image, sc["prompt_c"], job["size"], batch.BATCH_MODEL, batch.BATCH_QUALITY)
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}.png"
        cost = core._store(run_dir, fname, png, usage, job["size"], batch.BATCH_MODEL, batch.BATCH_QUALITY,
                           f"scene flow image | {folder}")
        with batch._LOCK:
            it.update(status="done", file=fname, cost_inr=round(cost, 4), error="")
            sc["stage"] = "done"
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
        lib.update_scene(sc["scene_id"], status="done", file=fname)
        batch._count(job)
        batch._write(folder, job)
    except Exception as e:
        _fail(folder, job, it, e)


def run_scene_flow(folder: str, indexes: list):
    """Runs (or continues) the products in `indexes`. Finished steps are kept, so this is also the retry."""
    job = batch.BJOBS[folder]
    try:
        todo = [i for i in indexes if job["items"][i]["status"] != "done"]
        # 1. identify every product that is not identified yet (in parallel, text calls are cheap)
        need = [i for i in todo if "category" not in job["items"][i]["scene"]]
        list(batch.BATCH_POOL.map(lambda i: _classify(folder, job, i), need))
        # 2. settle the names one by one: an existing spelling always wins, and a new name is registered at once,
        #    so the next photo of this batch (and the next run) files into it
        for i in need:
            it = job["items"][i]
            sc = it["scene"]
            if it["status"] == "error" or "raw_category" not in sc:
                continue
            cat, sub, new_cat, new_sub = lib.resolve(sc["raw_category"], sc["raw_subcategory"])
            lib.ensure(cat, sub)
            _set(sc, category=cat, subcategory=sub, new_category=new_cat, new_subcategory=new_sub)
        batch._write(folder, job)
        # 3. photos in the same subcategory run one after another; different subcategories run in parallel
        groups = {}
        for i in todo:
            sc = job["items"][i]["scene"]
            if job["items"][i]["status"] != "error" and sc.get("category"):
                groups.setdefault((sc["category"], sc["subcategory"]), []).append(i)

        def run_group(idxs):
            for i in idxs:
                _scene_and_image(folder, job, i)

        list(batch.BATCH_POOL.map(run_group, list(groups.values())))
    except Exception as e:                                # should not happen; never leave a run "running" forever
        _set(job, error=str(e))
    _set(job, state="done")
    batch._count(job)
    batch._write(folder, job)


def retry_scene_flow(folder: str, job: dict) -> int:
    """Re-run every product that is not done, each from the step that failed. Returns how many."""
    bad = [i for i, it in enumerate(job["items"]) if it["status"] != "done"]
    for i in bad:
        it = job["items"][i]
        if it["scene"].get("scene_id"):
            lib.update_scene(it["scene"]["scene_id"], status="pending")
        it.update(status="queued", error="")
    if not bad:
        return 0
    job.pop("error", None)
    job["state"] = "running"
    batch._count(job)
    batch.BJOBS[folder] = job
    threading.Thread(target=run_scene_flow, args=(folder, bad), daemon=True).start()
    return len(bad)


@router.post("/api/scene/run")
async def scene_run(
    images: list[UploadFile] = File(...),
    text_model: str = Form(...),
    effort_classify: str = Form("low"),
    effort_scene: str = Form("medium"),
    prompt_a1: str = Form(""),
    prompt_a2: str = Form(""),
    prompt_b: str = Form(""),
    size: str = Form("1088x1360"),
    name: str = Form(""),
):
    tm, prov, ready = _models()
    if text_model not in tm:
        raise HTTPException(400, "Pick a text model.")
    if not ready(text_model):
        raise HTTPException(400, f"{text_model} needs {'GEMINI_API_KEY' if prov[text_model] == 'google' else 'MODEL_API_KEY'} in the server's .env file.")
    for e in (effort_classify, effort_scene):
        if e not in tm[text_model]["efforts"]:
            raise HTTPException(400, f"{text_model} does not support effort '{e}'. Supported: {', '.join(tm[text_model]['efforts'])}.")
    a1, a2 = prompt_a1.strip() or DEFAULT_A1, prompt_a2.strip() or DEFAULT_A2
    if len(a1) > 30000 or len(a2) > 30000 or len(prompt_b) > 10000:
        raise HTTPException(400, "A prompt is too long.")
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > batch.BATCH_MAX:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {batch.BATCH_MAX} per run.")
    core._check_size(size)
    read = [await core._read_image(up) for up in images]
    d = batch._new_folder(name or "scene")
    folder = d.name
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "status": "queued", "error": "", "file": None, "cost_inr": 0,
                      "scene": {"stage": "queued", "input": fname}})
    job = {"folder": folder, "flow": "scene", "state": "running", "name": name.strip() or "Scene flow",
           "prompt": prompt_b.strip(), "size": size, "model": batch.BATCH_MODEL, "quality": batch.BATCH_QUALITY,
           "created": datetime.now().strftime("%d %b %Y, %H:%M"), "total": len(items), "done": 0, "total_inr": 0,
           "items": items,
           "scene_cfg": {"text_model": text_model, "effort_classify": effort_classify, "effort_scene": effort_scene,
                         "prompt_a1": a1, "prompt_a2": a2, "prompt_b": prompt_b.strip()}}
    batch.BJOBS[folder] = job
    batch._write(folder, job)
    threading.Thread(target=run_scene_flow, args=(folder, list(range(len(items)))), daemon=True).start()
    return {"folder": folder, "photos": len(items)}
