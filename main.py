import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from starlette.concurrency import run_in_threadpool

from costing import (META_MODELS, NANO_MODELS, NANO_SIZE_FOR_QUALITY, USD_TO_INR, _as_dict, compute_cost, log_cost, log_text_cost,
                     total_so_far)
from runs import ext_for, new_run_dir, write_comparison
from prompts import (
    MASTER_PROMPT, MASTER_STAGE_NOTE, STYLES, build_custom_prompt,
    build_preset_prompt, extract_winner_prompts,
)

load_dotenv()
COST_LOG = str(Path(__file__).parent / "cost_log.csv")
OUTPUT_DIR = Path(__file__).parent / "outputs"  # one numbered sub-folder per request

# Defaults used when a request does not send `model` / `quality`.
MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-flare")
QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "high")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-6.1-sol")  # reads the image + master prompt
MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}

app = FastAPI(title="Image Generation Backend")
client = OpenAI()  # reads OPENAI_API_KEY from the environment

OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")  # open saved images in the browser


# Dropdowns in /docs are built from these enums. Add a model here to allow it.
Style = Enum("Style", {s: s for s in STYLES}, type=str)


class Model(str, Enum):
    flare = "gpt-image-2.5-flare"        # fast, everyday
    sunburst = "gpt-image-2.5-sunburst"  # most capable, precise edits
    gpt_image_1 = "gpt-image-1"          # older model


class Quality(str, Enum):
    auto = "auto"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"  # 2.5 models only
    max = "max"      # 2.5 models only


class TextModel(str, Enum):
    sol = "gpt-6.1-sol"      # good balance (default)
    astra = "gpt-6-astra"    # most capable, costs more
    luna = "gpt-6-luna"      # cheapest


LEGACY_QUALITIES = {"auto", "low", "medium", "high"}  # what gpt-image-1 accepts


def _resolve(model: Model | None, quality: Quality | None) -> tuple[str, str]:
    """Use the request's model/quality, else the .env defaults; reject bad combos."""
    m = model.value if model else MODEL
    q = quality.value if quality else QUALITY
    if m == Model.gpt_image_1.value and q not in LEGACY_QUALITIES:
        raise HTTPException(400, f"Quality '{q}' is only available on the 2.5 models.")
    return m, q


def _check_size(size: str) -> None:
    """Catch invalid sizes before spending an API call (rules for the 2.5 models)."""
    if size == "auto":
        return
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError:
        raise HTTPException(400, "size must be 'auto' or WIDTHxHEIGHT, e.g. 1024x1024.")
    if w % 16 or h % 16:
        raise HTTPException(400, "Width and height must be multiples of 16.")
    if max(w, h) > 3840 or not (655_360 <= w * h <= 8_294_400):
        raise HTTPException(400, "Size out of range (max edge 3840, 655,360 to 8,294,400 pixels).")
    if max(w, h) / min(w, h) > 3:
        raise HTTPException(400, "Aspect ratio can be at most 3:1.")


async def _read_image(file: UploadFile) -> tuple[str, bytes, str]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Upload a PNG, JPEG or WEBP image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be under {MAX_UPLOAD_MB} MB.")
    return (file.filename or "input.png", data, file.content_type)


_NANO_RATIOS = [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9)]
_gemini = None
_gemini_lock = threading.Lock()
try:                                   # load Google's library once at start-up: importing it from several
    from google import genai as _genai  # worker threads at the same time gives a "circular import" error
except Exception:                      # not installed yet: only Nano Banana is unavailable
    _genai = None


def gemini_ready() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _gemini_client():
    global _gemini
    with _gemini_lock:
        if _gemini is None:
            if not gemini_ready():
                raise HTTPException(400, "Nano Banana needs GEMINI_API_KEY in the server's .env file.")
            if _genai is None:
                raise HTTPException(500, "Google's library is not installed: run pip install -r requirements.txt on the server and restart.")
            _gemini = _genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        return _gemini


def _aspect_for(size: str) -> str:
    """'1088x1360' -> '4:5' (the nearest ratio Nano Banana supports)."""
    w, h = (int(x) for x in size.lower().split("x"))
    a, b = min(_NANO_RATIOS, key=lambda r: abs(r[0] / r[1] - w / h))
    return f"{a}:{b}"


def _generate_nano(image, prompt: str, size: str, model: str, quality: str):
    """Nano Banana (Google) edit: input photo + prompt -> image bytes. quality low/medium/high -> 1K/2K/4K."""
    cfg = {}
    if os.getenv("NANO_THINKING"):                     # optional: minimal | medium | high
        cfg["generation_config"] = {"thinking_level": os.getenv("NANO_THINKING")}
    try:
        it = _gemini_client().interactions.create(
            model=model,
            input=[{"type": "text", "text": prompt},
                   {"type": "image", "data": base64.b64encode(image[1]).decode("ascii"), "mime_type": image[2] or "image/png"}],
            response_format={"type": "image", "aspect_ratio": _aspect_for(size),
                             "image_size": NANO_SIZE_FOR_QUALITY.get(quality, "2K")},
            **cfg,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Google error: {e}")
    data = getattr(getattr(it, "output_image", None), "data", None)
    if not data:
        why = (getattr(it, "output_text", None) or "").strip()[:200]
        raise HTTPException(502, "Google returned no image" + (f": {why}" if why else "."))
    u = getattr(it, "usage", None)
    usage = {"input_tokens": int(getattr(u, "total_input_tokens", 0) or 0),
             # thinking tokens are billed at the output rate
             "output_tokens": int(getattr(u, "total_output_tokens", 0) or 0) + int(getattr(u, "total_thought_tokens", 0) or 0)}
    return base64.b64decode(data), usage


_meta = None
_meta_lock = threading.Lock()


def meta_ready() -> bool:
    return bool(os.getenv("MODEL_API_KEY"))


def _meta_client():
    global _meta
    with _meta_lock:
        if _meta is None:
            if not meta_ready():
                raise HTTPException(400, "Muse Image needs MODEL_API_KEY in the server's .env file.")
            _meta = OpenAI(base_url=os.getenv("META_API_BASE", "https://api.meta.ai/v1"),
                           api_key=os.getenv("MODEL_API_KEY"), timeout=240)   # Meta's API speaks the OpenAI format
        return _meta


def _generate_muse(image, prompt: str, size: str, model: str, quality: str):
    """Meta Muse Image edit: input photo + prompt -> image bytes. Flat price, no quality levels."""
    c = _meta_client()
    base = dict(model=model, prompt=prompt, image=image, n=1, output_format="png")
    try:
        try:
            result = c.images.edit(size=size, **base)         # size only sets the aspect ratio for Muse
        except OpenAIError as e:
            if "size" not in str(e).lower():
                raise
            result = c.images.edit(**base)                    # this endpoint refused size: retry without it
    except OpenAIError as e:
        raise HTTPException(502, f"Meta error: {e}")
    item = result.data[0] if result.data else None
    b64 = getattr(item, "b64_json", None)
    if not b64:
        raise HTTPException(502, "Meta returned no image data.")
    return base64.b64decode(b64), None


def generate_text(provider: str, model: str, effort: str, instructions: str, user_text: str, image):
    """One text-model call that reads photo(s). provider: openai | google | meta.
    `image` is one (name, bytes, content-type) tuple or a list of them. user_text is sent as written.
    Returns (reply text, usage dict {"input_tokens", "output_tokens"}); output includes hidden thinking tokens."""
    images = image if isinstance(image, list) else [image]
    parts = [(base64.b64encode(im[1]).decode("ascii"), im[2] or "image/png") for im in images]
    if provider == "google":
        try:
            it = _gemini_client().interactions.create(
                model=model,
                input=[{"type": "text", "text": f"{instructions}\n\n{user_text}"}]
                      + [{"type": "image", "data": b64, "mime_type": mime} for b64, mime in parts],
                generation_config={"thinking_level": effort},
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Google error: {e}")
        u = getattr(it, "usage", None)
        usage = {"input_tokens": int(getattr(u, "total_input_tokens", 0) or 0),
                 "output_tokens": int(getattr(u, "total_output_tokens", 0) or 0) + int(getattr(u, "total_thought_tokens", 0) or 0)}
        return (getattr(it, "output_text", None) or ""), usage
    pics = [{"type": "input_image", "image_url": f"data:{mime};base64,{b64}"} for b64, mime in parts]
    if provider == "meta":
        c = _meta_client()
        msg = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}] + pics}]
        try:
            try:
                resp = c.responses.create(model=model, instructions=instructions, reasoning={"effort": effort}, input=msg)
            except OpenAIError as e:
                if "instruction" not in str(e).lower():
                    raise
                folded = [{"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": f"{instructions}\n\n{user_text}"}] + pics}]
                resp = c.responses.create(model=model, reasoning={"effort": effort}, input=folded)
        except OpenAIError as e:
            raise HTTPException(502, f"Meta error: {e}")
    else:
        try:
            resp = client.responses.create(model=model, instructions=instructions, reasoning={"effort": effort},
                                           input=[{"role": "user", "content": [{"type": "input_text", "text": user_text}] + pics}])
        except OpenAIError as e:
            raise HTTPException(502, f"OpenAI error: {e}")
    return (resp.output_text or ""), _as_dict(getattr(resp, "usage", None))


def _generate(image, prompt: str, size: str, model: str, quality: str):
    """Call the image model. Returns (png_bytes, usage). Nothing is saved here."""
    if model in NANO_MODELS:
        return _generate_nano(image, prompt, size, model, quality)
    if model in META_MODELS:
        return _generate_muse(image, prompt, size, model, quality)
    try:
        # `image` may be one tuple or a list of them (several photos of one product)
        result = client.images.edit(model=model, image=image, prompt=prompt, size=size, quality=quality)
    except OpenAIError as e:
        raise HTTPException(502, f"OpenAI error: {e}")
    return base64.b64decode(result.data[0].b64_json), getattr(result, "usage", None)


def _store(run_dir: Path, filename: str, png: bytes, usage, size: str, model: str, quality: str, note: str,
           ledger: list | None = None) -> float:
    """Save one generated image inside its run folder and log its cost. Returns cost in INR."""
    (run_dir / filename).write_bytes(png)
    cost, source, _ = log_cost(COST_LOG, model, quality, size, usage, note=f"{note} | {run_dir.name}/{filename}")
    if ledger is not None:
        _, _, tin, tout = compute_cost(usage, quality, model)
        ledger.append({"part": f"Image: {filename}", "model": model, "tokens_in": tin, "tokens_out": tout,
                       "cost_inr": cost, "source": source})
    return cost


def _save_input(run_dir: Path, img) -> str:
    name = "input" + ext_for(img[2])
    (run_dir / name).write_bytes(img[1])
    return name


def _single_run(img, prompt: str, size: str, m: str, q: str, label: str, title: str, note: str, shown_prompt: str,
                short_prompt: str = ""):
    """custom / preset: one image in, one image out, saved in its own numbered folder."""
    png, usage = _generate(img, prompt, size, m, q)
    run_dir = new_run_dir(OUTPUT_DIR, m, q)
    in_name = _save_input(run_dir, img)
    ledger = []
    cost = _store(run_dir, f"{label}.png", png, usage, size, m, q, note, ledger)
    write_comparison(
        run_dir, title, in_name,
        [{"label": label, "file": f"{label}.png", "cost_inr": cost, "prompt": shown_prompt}],
        {"Prompt": short_prompt or title, "Model": m, "Quality": q, "Size": size, "Cost": f"\u20b9{cost:.2f}"},
        costs=ledger,
    )
    return png, cost, run_dir.name


def _image_response(png: bytes, cost: float, model: str, quality: str, folder: str) -> Response:
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Cost-INR": f"{cost:.4f}", "X-Model": model, "X-Quality": quality, "X-Saved-In": f"outputs/{folder}"},
    )


@app.post("/generate/custom", response_class=Response)
async def generate_custom(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Model | None = Form(None, description="Defaults to OPENAI_IMAGE_MODEL"),
    quality: Quality | None = Form(None, description="Defaults to OPENAI_IMAGE_QUALITY"),
    size: str = Form("1024x1024"),
):
    """Mode 1: upload an image + a one-line prompt, get the image back."""
    m, q = _resolve(model, quality)
    _check_size(size)
    img = await _read_image(image)
    full = build_custom_prompt(prompt)
    png, cost, folder = await run_in_threadpool(
        _single_run, img, full, size, m, q, "custom", "Custom prompt", "custom", full, prompt.strip())
    return _image_response(png, cost, m, q, folder)


@app.post("/generate/preset", response_class=Response)
async def generate_preset(
    image: UploadFile = File(...),
    style: Style = Form(...),
    model: Model | None = Form(None, description="Defaults to OPENAI_IMAGE_MODEL"),
    quality: Quality | None = Form(None, description="Defaults to OPENAI_IMAGE_QUALITY"),
    size: str = Form("1024x1024"),
):
    """Mode 2: upload an image + pick a style. Prompt is hardcoded server-side."""
    m, q = _resolve(model, quality)
    _check_size(size)
    img = await _read_image(image)
    full = build_preset_prompt(style.value)
    png, cost, folder = await run_in_threadpool(
        _single_run, img, full, size, m, q, style.value, f"Preset: {style.value}", f"preset:{style.value}", full,
        f"Preset: {style.value} (built-in prompt)")
    return _image_response(png, cost, m, q, folder)


MASTER_LABELS = {"ai_model": "AI model ad", "lifestyle": "Lifestyle ad", "studio": "Studio ad"}


def _run_master(img, size: str, m: str, q: str, text_model: str) -> dict:
    """Step A: text model runs the master prompt. Step B: three separate image edits."""
    b64 = base64.b64encode(img[1]).decode()
    kwargs = dict(
        model=text_model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": MASTER_PROMPT},
            {"type": "input_image", "image_url": f"data:{img[2]};base64,{b64}"},
        ]}],
    )
    if MASTER_STAGE_NOTE:
        kwargs["instructions"] = MASTER_STAGE_NOTE
    try:
        resp = client.responses.create(**kwargs)
    except OpenAIError as e:
        raise HTTPException(502, f"OpenAI error (analysis step): {e}")
    text = resp.output_text or ""
    text_cost, text_src, _ = log_text_cost(COST_LOG, text_model, getattr(resp, "usage", None), note="master:analysis")
    _tu = _as_dict(getattr(resp, "usage", None))
    ledger = [{"part": "Analysis (text model reads the image + master prompt)", "model": text_model,
               "tokens_in": int(_tu.get("input_tokens") or 0), "tokens_out": int(_tu.get("output_tokens") or 0),
               "cost_inr": text_cost, "source": text_src}]

    # Everything for this request goes in one numbered folder.
    run_dir = new_run_dir(OUTPUT_DIR, m, q)
    in_name = _save_input(run_dir, img)
    (run_dir / "analysis.md").write_text(text, encoding="utf-8")
    try:
        winners = extract_winner_prompts(text)
    except ValueError as e:
        raise HTTPException(502, f"Could not read the three winning prompts ({e}). "
                                 f"The full reply was saved as outputs/{run_dir.name}/analysis.md.")

    def one(style: str, prompt: str):
        try:
            png, usage = _generate(img, prompt, size, m, q)
            return style, png, usage, None
        except HTTPException as e:
            return style, None, None, str(e.detail)

    with ThreadPoolExecutor(max_workers=3) as pool:
        done = list(pool.map(lambda kv: one(*kv), winners.items()))

    images, errors, items = [], {}, []
    for style, png, usage, err in done:
        label = MASTER_LABELS[style]
        if png is None:
            errors[style] = err
            items.append({"label": label, "file": None, "error": err, "prompt": winners[style]})
            continue
        cost = _store(run_dir, f"{style}.png", png, usage, size, m, q, f"master:{style}", ledger)
        images.append({"style": style, "file": f"{style}.png", "url": f"/outputs/{run_dir.name}/{style}.png",
                       "cost_inr": round(cost, 2)})
        items.append({"label": label, "file": f"{style}.png", "cost_inr": cost, "prompt": winners[style]})

    total = sum(c["cost_inr"] for c in ledger)
    write_comparison(
        run_dir, "Master prompt: input vs. three generated ads", in_name, items,
        {"Prompt": "Master prompt (AI model + lifestyle + studio ads)",
         "Image model": m, "Quality": q, "Size": size, "Text model": text_model,
         "Total cost": f"\u20b9{total:.2f}"},
        analysis=text, costs=ledger,
    )
    if not images:
        raise HTTPException(502, f"All three image generations failed: {errors}. "
                                 f"The analysis was saved in outputs/{run_dir.name}/.")
    return {
        "folder": run_dir.name,
        "comparison_url": f"/outputs/{run_dir.name}/comparison.html",
        "images": images,
        "errors": errors,
        "winning_prompts": winners,
        "text_model": text_model, "image_model": m, "quality": q,
        "analysis_cost_inr": round(text_cost, 2),
        "total_cost_inr": round(total, 2),
    }


@app.post("/generate/master")
async def generate_master(
    image: UploadFile = File(...),
    model: Model | None = Form(None, description="Image model. Defaults to OPENAI_IMAGE_MODEL"),
    quality: Quality | None = Form(None, description="Image quality. Defaults to OPENAI_IMAGE_QUALITY"),
    text_model: TextModel | None = Form(None, description="Model that reads the image. Defaults to OPENAI_TEXT_MODEL"),
    size: str = Form("1088x1360", description="4:5 portrait"),
):
    """Master prompt: upload ONE image, get THREE separate ads (AI model, lifestyle, studio)."""
    m, q = _resolve(model, quality)
    _check_size(size)
    img = await _read_image(image)
    return await run_in_threadpool(_run_master, img, size, m, q, text_model.value if text_model else TEXT_MODEL)


@app.get("/cost")
def cost():
    """Total spent through this backend, from cost_log.csv."""
    return {"total_inr": round(total_so_far(COST_LOG), 2), "log_file": COST_LOG}


@app.get("/health")
def health():
    return {
        "ok": True,
        "default_model": MODEL,
        "default_quality": QUALITY,
        "default_text_model": TEXT_MODEL,
        "models": [m.value for m in Model],
        "qualities": [q.value for q in Quality],
    }


# ---- web page + /api/* used by the browser UI (see webapp.py) ----
from webapp import PasswordMiddleware, router as web_router  # noqa: E402  (needs the helpers above)

app.add_middleware(PasswordMiddleware)
app.include_router(web_router)
from batch import router as batch_router  # noqa: E402  (batch mode: one prompt x many images, own review sheet)
app.include_router(batch_router)
from sceneflow import router as scene_router  # noqa: E402  (batch scene flow + scene library)
app.include_router(scene_router)
