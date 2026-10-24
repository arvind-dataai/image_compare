"""Batch "scene flow", run step by step. Every uploaded photo is its OWN product.

  GET  /api/scene/options          text models, default prompts A1 / A2, sizes
  POST /api/scene/create           photos + name -> a run waiting for step 1, returns {folder}
  POST /api/scene/{folder}/step1   Identify : prompt A1 + text model + effort        -> category, subcategory, how many scenes exist
  POST /api/scene/{folder}/step2   Scene    : prompt A2 + text model + effort        -> a NEW structured scene (compared with the earlier ones)
  POST /api/scene/{folder}/step3   Image    : Prompt B ({SCENE} is replaced by the scene = Prompt C) -> gpt-image-2.5-sunburst, low
  POST /api/scene/{folder}/category  change the category / subcategory chosen in step 1 (clears the later steps of that product)
  POST /api/scene/{folder}/select    choose which scene run of a product step 3 uses (all runs stay in the history)
  GET  /api/scene/{folder}/promptc   preview of Prompt C for a given Prompt B
  GET  /api/scene/library          category > subcategory > scenes (for the Scene library tab)

Each step can be run for all products (default) or for chosen ones (`indexes`); re-running a step clears that product's later
steps. Step 2 does not use Prompt B. Photos that land in the same subcategory are done one after the other in step 2, so every new
scene sees the scenes just written for the earlier photos; different subcategories run in parallel.

Stored like any batch (outputs/batch_<n>_<name>/batch.json, flow = "scene"), so the Batch list, Batch review and its Excel handle it.
Per product, item["scene"]["step"] = last finished step (0-3); the scenes themselves live in scene_library.json (scene_library.py).
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
    '{"category": "...", "subcategory": "...", "product_description": "two or three plain sentences about the product as described above", '
    '"reason": "one or two sentences: why it belongs there and which existing option you considered closest"}'
)
SCENE_FORMAT = (
    'Reply with ONE JSON object and nothing else (no markdown, no text before or after). Every value is a string with real detail '
    '(one to three sentences), except where a list is shown:\n'
    '{\n'
    '  "concept_name": "a short name for this scene",\n'
    '  "setting_environment": "the exact place and the spot within it, with the time of day",\n'
    '  "surface_and_background": "the surface under the product, the mid-ground and the far background, each a real material with its colour",\n'
    '  "product_presentation": "the most important face of the product to the camera, at true size, and how it sits, hangs or stands (product named by type only)",\n'
    '  "composition": "where the product sits in the 4:5 frame, the calm margin around it, and the side left as clean space",\n'
    '  "camera_angle_and_framing": "camera angle and distance",\n'
    '  "lighting": "light direction and quality, product lit a step brighter, soft contact shadow",\n'
    '  "colour_palette": "dominant tone, secondary tone and one accent, then the required palette sentence",\n'
    '  "props": "each prop with its position beside or behind the product and out of focus, or none",\n'
    '  "mood_and_style": "the feeling and photographic style",\n'
    '  "difference_from_previous": "how this scene differs from the previous scenes (name them)"\n'
    '}'
)


def _models():
    from webapp import TEXT_MODELS, TEXT_PROVIDER, _text_ready
    return TEXT_MODELS, TEXT_PROVIDER, _text_ready


def _est_image():
    from costing import estimate_inr
    return round(estimate_inr(batch.BATCH_QUALITY, batch.BATCH_MODEL), 4)


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
        "image_model": batch.BATCH_MODEL, "quality": batch.BATCH_QUALITY, "est_image_inr_1024": _est_image(),
    }


@router.get("/api/scene/library")
def scene_library_view():
    return {"categories": lib.overview()}


# ---------------------------------------------------------------- state helpers
S1_KEYS = ["raw_category", "raw_subcategory", "category", "subcategory", "new_category", "new_subcategory",
           "product_description", "classify_reason", "existing_scenes", "s1_cfg", "edited"]
S2_KEYS = ["scene", "scene_description", "previous_count", "previous", "s2_cfg", "scene_id"]
S3_KEYS = ["prompt_c", "prompt_b", "s3_cfg"]


def sync(it: dict):
    """Item status (used by the batch list and the review tab) from its step state. Call with batch._LOCK held."""
    sc = it["scene"]
    if sc.get("running"):
        it["status"], it["error"] = "running", ""
    elif sc.get("error"):
        it["status"], it["error"] = "error", sc["error"]
    elif it.get("file"):
        it["status"], it["error"] = "done", ""
    else:
        it["status"], it["error"] = "queued", ""


def _now() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


def _migrate(it: dict):
    """Every scene run (step 2) and every image run (step 3) of a product is kept in two lists, with its own cost.
    Runs saved by an older version only kept the latest scene and image: turn those into list entries too."""
    sc = it["scene"]
    sc.setdefault("runs2", [])
    sc.setdefault("runs3", [])
    if not sc["runs2"] and sc.get("scene"):
        c = sc.get("s2_cfg") or {}
        sc["runs2"].append({"n": 1, "scene_id": sc.get("scene_id"), "created": "", "category": sc.get("category"),
                            "subcategory": sc.get("subcategory"), "text_model": c.get("text_model", ""), "effort": c.get("effort", ""),
                            "prompt_a2": "", "previous_count": sc.get("previous_count", 0), "previous": sc.get("previous", []),
                            "scene": sc["scene"], "scene_description": sc.get("scene_description", ""),
                            "cost_inr": sc.get("scene_cost_inr", 0)})
        sc["sel2"] = 1
    if not sc["runs3"] and it.get("file") and sc.get("scene"):
        sc["runs3"].append({"n": 1, "scene_n": sc.get("sel2", 1), "scene_id": sc.get("scene_id"), "created": "",
                            "prompt_b": sc.get("prompt_b", ""), "prompt_c": sc.get("prompt_c", ""),
                            "size": (sc.get("s3_cfg") or {}).get("size", ""), "file": it["file"], "cost_inr": it.get("cost_inr", 0)})


def _apply_selection(sc: dict, rec: dict, has_image: bool):
    """Make one scene run the selected one, the scene that step 3 draws from. Call with batch._LOCK held."""
    sc.update(scene=rec["scene"], scene_description=rec["scene_description"], scene_id=rec["scene_id"],
              previous_count=rec.get("previous_count", 0), previous=rec.get("previous", []),
              s2_cfg={"text_model": rec.get("text_model", ""), "effort": rec.get("effort", "")},
              sel2=rec["n"], step=3 if has_image else 2)


def _unselect(it: dict):
    """The product has no current scene any more (step 1 was run again, or its category was changed).
    Its earlier scene and image runs stay in the history. Call with batch._LOCK held."""
    sc = it["scene"]
    if sc.get("scene_id"):
        lib.update_scene(sc["scene_id"], status="replaced")          # no longer counts as "used"
    for k in S2_KEYS + S3_KEYS:
        sc.pop(k, None)
    sc["sel2"] = None


def _job(folder: str) -> dict:
    batch._dir(folder)
    job = batch._load(folder)
    batch.BJOBS[folder] = job
    if job.get("flow") != "scene":
        raise HTTPException(400, "This is not a scene-flow run.")
    return job


def _eligible(job: dict, step: int, indexes: list) -> list:
    out = []
    for i in indexes:
        if not (0 <= i < len(job["items"])):
            continue
        sc = job["items"][i]["scene"]
        if sc.get("running"):
            continue
        if sc.get("step", 0) >= step - 1:                  # step 1 always; step 2 needs 1 done; step 3 needs 2 done
            out.append(i)
    return out


def _begin(folder: str, step: int, indexes_raw: str) -> tuple:
    job = _job(folder)
    with batch._LOCK:
        if job.get("state") == "running":
            raise HTTPException(400, "A step is still running. Wait for it to finish.")
        wanted = ([int(x) for x in indexes_raw.split(",") if x.strip().isdigit()] if indexes_raw.strip()
                  else list(range(len(job["items"]))))
        todo = _eligible(job, step, wanted)
        if not todo:
            raise HTTPException(400, {1: "Nothing to run.", 2: "Run step 1 first: no product has been identified yet.",
                                      3: "Run step 2 first: no product has a scene yet."}[step])
        job["state"] = "running"
        job.pop("error", None)
        for i in todo:
            it = job["items"][i]
            sc = it["scene"]
            _migrate(it)
            if step == 1:                                   # a new identification: the earlier scene / image runs stay in the history
                for k in S1_KEYS:
                    sc.pop(k, None)
                _unselect(it)
                sc["step"] = 0
            sc["error"], sc["error_step"] = "", 0
            sc["running"] = step
            sync(it)
    batch._write(folder, job)
    return job, todo


def _finish(folder: str, job: dict, todo: list):
    with batch._LOCK:
        for i in todo:
            job["items"][i]["scene"]["running"] = 0
            sync(job["items"][i])
        job["state"] = "ready"
        batch._count(job)
    batch._write(folder, job)


def _fail(folder: str, job: dict, i: int, e: Exception):
    it = job["items"][i]
    detail = str(getattr(e, "detail", None) or e)[:600]
    with batch._LOCK:
        it["scene"]["error"] = detail
        it["scene"]["error_step"] = it["scene"].get("running") or 0
        it["scene"]["running"] = 0
        sync(it)
    batch._write(folder, job)


def _set(d: dict, **kw):
    with batch._LOCK:                                   # never change a dict while batch.json is being written
        d.update(kw)


def _text_call(job: dict, sc: dict, label: str, model: str, effort: str, instructions: str, user_text: str, image, cost_key: str):
    """One text call + its cost. Retries once if the reply is not usable JSON. Returns the parsed dict."""
    provider = _models()[1][model]
    last_err = ""
    for attempt in (1, 2):
        extra = "" if attempt == 1 else (f"\n\nYour previous reply could not be used ({last_err}). "
                                         "Reply again with ONLY the JSON object described above.")
        text, usage = core.generate_text(provider, model, effort, instructions, user_text + extra, image)
        cost, _, _ = log_text_cost(core.COST_LOG, model, usage,
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


# ---------------------------------------------------------------- step 1: identify
def _do_step1(folder: str, job: dict, i: int, cfg: dict):
    it = job["items"][i]
    sc = it["scene"]
    try:
        instr = (cfg["prompt"].strip() + "\n\nKNOWN TAXONOMY (categories and their subcategories already in the library):\n"
                 + lib.taxonomy_text() + "\n\n" + CLASSIFY_FORMAT)
        r = _text_call(job, sc, "identify", cfg["text_model"], cfg["effort"], instr,
                       "Identify the single product shown in the attached photo and file it, as instructed.",
                       _image_of(folder, it), "classify_cost_inr")
        _need(r, lib.CLASSIFY_KEYS, "identification")
        _set(sc, raw_category=str(r["category"]), raw_subcategory=str(r["subcategory"]),
             product_description=str(r["product_description"]).strip(), classify_reason=str(r.get("reason", "")).strip(),
             s1_cfg={"text_model": cfg["text_model"], "effort": cfg["effort"]})
    except Exception as e:
        _fail(folder, job, i, e)


def _run_step1(folder: str, job: dict, todo: list, cfg: dict):
    try:
        list(batch.BATCH_POOL.map(lambda i: _do_step1(folder, job, i, cfg), todo))
        # settle the names one by one: an existing spelling always wins, a new name is registered at once,
        # so the next photo of this run (and every later run) files into it
        for i in todo:
            sc = job["items"][i]["scene"]
            if sc.get("error") or "raw_category" not in sc:
                continue
            cat, sub, new_cat, new_sub = lib.resolve(sc["raw_category"], sc["raw_subcategory"])
            lib.ensure(cat, sub)
            _set(sc, category=cat, subcategory=sub, new_category=new_cat, new_subcategory=new_sub,
                 existing_scenes=len(lib.previous(cat, sub, 10 ** 6)), step=1)
    except Exception as e:
        for i in todo:
            if job["items"][i]["scene"].get("step", 0) < 1 and not job["items"][i]["scene"].get("error"):
                _fail(folder, job, i, e)
    _finish(folder, job, todo)


# ---------------------------------------------------------------- step 2: scene
def _do_step2(folder: str, job: dict, i: int, cfg: dict):
    it = job["items"][i]
    sc = it["scene"]
    before = sc.get("scene_cost_inr", 0)
    n = len(sc.get("runs2", [])) + 1
    base = {"n": n, "created": _now(), "category": sc.get("category"), "subcategory": sc.get("subcategory"),
            "text_model": cfg["text_model"], "effort": cfg["effort"], "prompt_a2": cfg["prompt"]}
    try:
        own = {r.get("scene_id") for r in sc.get("runs2", [])}      # this product's own earlier scene runs do not count as "used"
        prev = [p for p in lib.previous(sc["category"], sc["subcategory"], 10 ** 6) if p.get("id") not in own][-PREV_LIMIT:]
        prev_txt = ("\n".join(f"{k}. {p.get('summary') or lib.summary_of(p.get('scene', {}))}" for k, p in enumerate(prev, 1))
                    if prev else "None yet: this is the first scene for this subcategory.")
        ctx = (f"CONTEXT\nPRODUCT\ncategory: {sc['category']}\nsubcategory: {sc['subcategory']}\ndescription: {sc['product_description']}\n\n"
               f"PREVIOUS SCENES ALREADY USED FOR THIS SUBCATEGORY ({len(prev)}):\n{prev_txt}")
        instr = cfg["prompt"].strip() + "\n\n" + ctx + "\n\n" + SCENE_FORMAT
        r = _text_call(job, sc, "scene", cfg["text_model"], cfg["effort"], instr,
                       "Design the new scene for the product in the attached photo, as instructed.",
                       _image_of(folder, it), "scene_cost_inr")
        _need(r, lib.SCENE_KEYS, "scene")
        scene = {k: r[k] for k in lib.SCENE_KEYS}
        scene["difference_from_previous"] = str(r.get("difference_from_previous", "")).strip()
        desc = lib.render_scene(sc["category"], sc["subcategory"], sc["product_description"], scene)
        sid = lib.add_scene(sc["category"], sc["subcategory"], {
            "status": "pending", "folder": folder, "input": it["input"], "text_model": cfg["text_model"],
            "effort": cfg["effort"], "scene": scene, "scene_description": desc, "summary": lib.summary_of(scene)})
        rec = {**base, "scene_id": sid, "previous_count": len(prev), "scene": scene, "scene_description": desc,
               "previous": [{"concept": (p.get("scene") or {}).get("concept_name", ""), "summary": p.get("summary", "")} for p in prev],
               "cost_inr": round(sc.get("scene_cost_inr", 0) - before, 4)}
        with batch._LOCK:
            old = sc.get("scene_id")
            sc.setdefault("runs2", []).append(rec)
            _apply_selection(sc, rec, False)                 # the newest scene run is the selected one
        if old and old != sid:
            lib.update_scene(old, status="replaced")         # only the selected scene of a product counts as "used"
    except Exception as e:
        cost = round(sc.get("scene_cost_inr", 0) - before, 4)
        if cost > 0:                                         # a failed call that still cost money stays in the history
            with batch._LOCK:
                sc.setdefault("runs2", []).append({**base, "error": str(getattr(e, "detail", None) or e)[:300], "cost_inr": cost})
        _fail(folder, job, i, e)


def _run_step2(folder: str, job: dict, todo: list, cfg: dict):
    try:
        groups = {}
        for i in todo:                                   # same subcategory: one after the other
            sc = job["items"][i]["scene"]
            groups.setdefault((sc["category"], sc["subcategory"]), []).append(i)

        def run_group(idxs):
            for i in idxs:
                _do_step2(folder, job, i, cfg)

        list(batch.BATCH_POOL.map(run_group, list(groups.values())))
    except Exception as e:
        for i in todo:
            if job["items"][i]["scene"].get("step", 0) < 2 and not job["items"][i]["scene"].get("error"):
                _fail(folder, job, i, e)
    _finish(folder, job, todo)


# ---------------------------------------------------------------- step 3: image
def _do_step3(folder: str, job: dict, i: int, prompt_b: str, size: str):
    it = job["items"][i]
    sc = it["scene"]
    try:
        prompt_c = lib.build_prompt_c(sc["scene_description"], prompt_b)
        png, usage = core._generate(_image_of(folder, it), prompt_c, size, batch.BATCH_MODEL, batch.BATCH_QUALITY)
        n3 = len(sc.get("runs3", [])) + 1
        k = max(sc.get("image_runs", 0), n3 - 1) + 1
        fname = f"out_{i + 1:03d}_{batch._safe_name(it['orig_name'])}_{k}.png"
        cost = core._store(core.OUTPUT_DIR / folder, fname, png, usage, size, batch.BATCH_MODEL, batch.BATCH_QUALITY,
                           f"scene flow image | {folder}")
        rec = {"n": n3, "scene_n": sc.get("sel2"), "scene_id": sc.get("scene_id"), "created": _now(), "prompt_b": prompt_b,
               "prompt_c": prompt_c, "size": size, "file": fname, "cost_inr": round(cost, 4)}
        with batch._LOCK:
            sc.setdefault("runs3", []).append(rec)
            it.update(file=fname, cost_inr=round((it.get("cost_inr") or 0) + cost, 4))
            sc.update(step=3, image_runs=k, prompt_c=prompt_c, prompt_b=prompt_b, s3_cfg={"size": size})
            job["total_inr"] = round(job.get("total_inr", 0) + cost, 4)
        lib.update_scene(sc["scene_id"], status="done", file=fname, prompt_b=prompt_b)
    except Exception as e:
        if sc.get("scene_id") and not any(r.get("scene_id") == sc["scene_id"] for r in sc.get("runs3", [])):
            lib.update_scene(sc["scene_id"], status="failed")
        _fail(folder, job, i, e)


def _run_step3(folder: str, job: dict, todo: list, prompt_b: str, size: str):
    try:
        list(batch.BATCH_POOL.map(lambda i: _do_step3(folder, job, i, prompt_b, size), todo))
    except Exception as e:
        for i in todo:
            if job["items"][i]["scene"].get("step", 0) < 3 and not job["items"][i]["scene"].get("error"):
                _fail(folder, job, i, e)
    with batch._LOCK:
        job["size"] = size
    _finish(folder, job, todo)


# ---------------------------------------------------------------- API
def _check_model(text_model: str, effort: str):
    tm, prov, ready = _models()
    if text_model not in tm:
        raise HTTPException(400, "Pick a text model.")
    if not ready(text_model):
        raise HTTPException(400, f"{text_model} needs {'GEMINI_API_KEY' if prov[text_model] == 'google' else 'MODEL_API_KEY'} in the server's .env file.")
    if effort not in tm[text_model]["efforts"]:
        raise HTTPException(400, f"{text_model} does not support effort '{effort}'. Supported: {', '.join(tm[text_model]['efforts'])}.")


@router.post("/api/scene/create")
async def scene_create(images: list[UploadFile] = File(...), name: str = Form("")):
    if not images:
        raise HTTPException(400, "Add at least one photo.")
    if len(images) > batch.BATCH_MAX:
        raise HTTPException(400, f"Too many photos ({len(images)}). The limit is {batch.BATCH_MAX} per run.")
    read = [await core._read_image(up) for up in images]
    d = batch._new_folder(name or "scene")
    folder = d.name
    items = []
    for n, img in enumerate(read, start=1):
        ext = ".png" if img[2] == "image/png" else ".webp" if img[2] == "image/webp" else ".jpg"
        fname = f"in_{n:03d}_{batch._safe_name(img[0])}{ext}"
        (d / fname).write_bytes(img[1])
        items.append({"input": fname, "orig_name": img[0], "status": "queued", "error": "", "file": None, "cost_inr": 0,
                      "scene": {"step": 0, "running": 0, "error": "", "input": fname, "runs2": [], "runs3": [], "sel2": None}})
    job = {"folder": folder, "flow": "scene", "state": "ready", "name": name.strip() or "Scene flow", "prompt": "",
           "size": "1088x1360", "model": batch.BATCH_MODEL, "quality": batch.BATCH_QUALITY,
           "created": datetime.now().strftime("%d %b %Y, %H:%M"), "total": len(items), "done": 0, "total_inr": 0, "items": items}
    batch.BJOBS[folder] = job
    batch._write(folder, job)
    return {"folder": folder, "photos": len(items)}


@router.post("/api/scene/{folder}/step1")
def scene_step1(folder: str, prompt_a1: str = Form(""), text_model: str = Form(...), effort: str = Form(...), indexes: str = Form("")):
    _check_model(text_model, effort)
    a1 = prompt_a1.strip() or DEFAULT_A1
    if len(a1) > 30000:
        raise HTTPException(400, "Prompt A1 is too long.")
    job, todo = _begin(folder, 1, indexes)
    threading.Thread(target=_run_step1, args=(folder, job, todo, {"prompt": a1, "text_model": text_model, "effort": effort}), daemon=True).start()
    return {"folder": folder, "running": len(todo)}


@router.post("/api/scene/{folder}/step2")
def scene_step2(folder: str, prompt_a2: str = Form(""), text_model: str = Form(...), effort: str = Form(...), indexes: str = Form("")):
    _check_model(text_model, effort)
    a2 = prompt_a2.strip() or DEFAULT_A2
    if len(a2) > 30000:
        raise HTTPException(400, "Prompt A2 is too long.")
    job, todo = _begin(folder, 2, indexes)
    threading.Thread(target=_run_step2, args=(folder, job, todo, {"prompt": a2, "text_model": text_model, "effort": effort}), daemon=True).start()
    return {"folder": folder, "running": len(todo)}


@router.post("/api/scene/{folder}/step3")
def scene_step3(folder: str, prompt_b: str = Form(""), size: str = Form("1088x1360"), indexes: str = Form("")):
    core._check_size(size)
    if len(prompt_b) > 10000:
        raise HTTPException(400, "Prompt B is too long.")
    job, todo = _begin(folder, 3, indexes)
    threading.Thread(target=_run_step3, args=(folder, job, todo, prompt_b.strip(), size), daemon=True).start()
    return {"folder": folder, "running": len(todo)}


@router.post("/api/scene/{folder}/category")
def scene_category(folder: str, payload: dict):
    """Change the category / subcategory chosen in step 1 for one product. Its scene (step 2) and image are cleared."""
    job = _job(folder)
    try:
        i = int(payload.get("index"))
        it = job["items"][i]
    except Exception:
        raise HTTPException(404, "Product not found.")
    cat, sub = str(payload.get("category", "")).strip(), str(payload.get("subcategory", "")).strip()
    if not cat or not sub:
        raise HTTPException(400, "Enter both a category and a subcategory.")
    with batch._LOCK:
        if job.get("state") == "running" or it["scene"].get("running"):
            raise HTTPException(400, "A step is running. Wait for it to finish.")
        if it["scene"].get("step", 0) < 1:
            raise HTTPException(400, "Run step 1 first.")
        _migrate(it)
        _unselect(it)
        cat, sub, new_cat, new_sub = lib.resolve(cat, sub)
        lib.ensure(cat, sub)
        it["scene"].update(category=cat, subcategory=sub, new_category=new_cat, new_subcategory=new_sub, edited=True,
                           existing_scenes=len(lib.previous(cat, sub, 10 ** 6)), step=1)
        sync(it)
    batch._write(folder, job)
    return {"ok": True}


@router.post("/api/scene/{folder}/select")
def scene_select(folder: str, payload: dict):
    """Choose which scene run of a product step 3 should use. Nothing is deleted: every run stays in the history."""
    job = _job(folder)
    try:
        i, n = int(payload.get("index")), int(payload.get("n"))
        it = job["items"][i]
    except Exception:
        raise HTTPException(404, "Product not found.")
    with batch._LOCK:
        if job.get("state") == "running" or it["scene"].get("running"):
            raise HTTPException(400, "A step is running. Wait for it to finish.")
        _migrate(it)
        sc = it["scene"]
        rec = next((r for r in sc["runs2"] if r["n"] == n and r.get("scene")), None)
        if not rec:
            raise HTTPException(404, "Scene run not found.")
        if (rec.get("category"), rec.get("subcategory")) != (sc.get("category"), sc.get("subcategory")):
            raise HTTPException(400, f"That scene was written for {rec.get('category')} > {rec.get('subcategory')}, "
                                     f"but this product is now filed as {sc.get('category')} > {sc.get('subcategory')}.")
        old = sc.get("scene_id")
        has_image = any(r.get("scene_n") == n for r in sc["runs3"])
        _apply_selection(sc, rec, has_image)
        sync(it)
    if old and old != rec["scene_id"]:
        lib.update_scene(old, status="replaced")
    lib.update_scene(rec["scene_id"], status="done" if has_image else "pending")
    batch._write(folder, job)
    return {"ok": True}


@router.get("/api/scene/{folder}/promptc")
def scene_promptc(folder: str, prompt_b: str = ""):
    job = _job(folder)
    return {"items": [{"index": i, "name": it["orig_name"], "prompt_c": lib.build_prompt_c(it["scene"]["scene_description"], prompt_b)}
                      for i, it in enumerate(job["items"]) if it["scene"].get("scene_description")]}
