"""Cost tracking for GPT Image 2.5 calls, reported in Indian rupees (INR).

Cost is computed from the token usage OpenAI returns with each image, using the
per-1M-token USD rates on OpenAI's model page, then converted to rupees with
USD_TO_INR. If a response has no usage data, a rough per-quality estimate for
1024x1024 is used and marked "estimate". The real source of truth is your
OpenAI Usage page (which bills in USD).

Set the exchange rate in .env:  USD_TO_INR=96.64
"""
import csv
import os
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Rupees per 1 US dollar. Override with USD_TO_INR in .env when the rate moves.
USD_TO_INR = float(os.getenv("USD_TO_INR", "96.64"))

# USD per 1,000,000 tokens (same for flare and sunburst)
RATES_PER_M = {
    "text_in": 5.00,
    "image_in": 8.00,
    "image_out": 30.00,
}

# Rough USD per image at 1024x1024 (third-party estimate; used only as a fallback)
ESTIMATE_1024 = {
    "low": 0.00588,
    "medium": 0.01317,
    "high": 0.05268,
    "xhigh": 0.09366,
    "max": 0.21072,
    "auto": 0.05268,
}

_LOG_LOCK = threading.Lock()   # several images/steps can finish at once; keep the running total correct

LOG_FIELDS = [
    "time", "step", "model", "quality", "size", "input_tokens", "output_tokens",
    "cost_inr", "cost_source", "running_total_inr", "usd_to_inr", "note",
]


# Google Nano Banana (Gemini image) models: USD per 1M tokens, and USD per image by output size
# (Gemini API pricing page). The app maps quality low/medium/high -> 1K/2K/4K.
NANO_MODELS = {
    "gemini-nano-banana-2.1": {"in": 1.50, "out": 30.00, "per_image": {"1K": 0.0336, "2K": 0.0504, "4K": 0.113}},
}
NANO_SIZE_FOR_QUALITY = {"low": "1K", "medium": "2K", "high": "4K"}


def estimate_inr(quality, model=None):
    """Rough rupee estimate for one image (1024x1024 for OpenAI models; the tier price for Nano Banana)."""
    if model in NANO_MODELS:
        tier = NANO_SIZE_FOR_QUALITY.get(quality, "2K")
        return NANO_MODELS[model]["per_image"][tier] * USD_TO_INR
    return ESTIMATE_1024.get(quality, ESTIMATE_1024["high"]) * USD_TO_INR


def _as_dict(usage):
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    try:
        return usage.model_dump()
    except Exception:
        return {}


def compute_cost(usage, quality="high", model=None):
    """Return (cost_inr, source, input_tokens, output_tokens)."""
    u = _as_dict(usage)
    if model in NANO_MODELS:                    # usage was normalised by main._generate_nano
        r = NANO_MODELS[model]
        in_tok, out_tok = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        if in_tok or out_tok:
            return (in_tok * r["in"] + out_tok * r["out"]) / 1_000_000 * USD_TO_INR, "usage", in_tok, out_tok
        return estimate_inr(quality, model), "estimate", 0, 0
    in_tok = int(u.get("input_tokens") or 0)
    out_tok = int(u.get("output_tokens") or 0)
    if in_tok or out_tok:
        d = u.get("input_tokens_details") or {}
        text_in = int(d.get("text_tokens") or 0)
        image_in = int(d.get("image_tokens") or 0)
        if not (text_in or image_in):          # no breakdown: bill all as image input
            image_in = in_tok
        usd = (
            text_in * RATES_PER_M["text_in"]
            + image_in * RATES_PER_M["image_in"]
            + out_tok * RATES_PER_M["image_out"]
        ) / 1_000_000
        return usd * USD_TO_INR, "usage", in_tok, out_tok
    return estimate_inr(quality), "estimate", 0, 0


def total_so_far(log_path):
    p = Path(log_path)
    if not p.is_file():
        return 0.0
    total = 0.0
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row.get("cost_inr") or 0)
            except ValueError:
                pass
    return total


def _archive_old_log(p):
    """Older logs lack the 'step' column: add it in place (text rows -> analysis, others -> image).
    A log with some other layout (e.g. the very old USD version) is set aside."""
    if not p.is_file():
        return
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == LOG_FIELDS:
        return
    header = rows[0]
    if header == [h for h in LOG_FIELDS if h != "step"]:
        out = []
        for r in rows[1:]:
            d = dict(zip(header, r))
            d["step"] = "analysis" if d.get("quality") == "text" else "image"
            out.append(d)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(out)
        return
    p.rename(p.with_name(f"{p.stem}_old_{datetime.now():%Y%m%d%H%M%S}{p.suffix}"))


def log_cost(log_path, model, quality, size, usage, note=""):
    """Append one row to the cost log. Returns (cost_inr, source, running_total_inr)."""
    cost, source, in_tok, out_tok = compute_cost(usage, quality, model)
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        _archive_old_log(p)
        running = total_so_far(p) + cost
        new_file = not p.is_file()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "time": datetime.now().isoformat(timespec="seconds"), "step": "image",
                "model": model, "quality": quality, "size": size,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_inr": f"{cost:.4f}", "cost_source": source,
                "running_total_inr": f"{running:.4f}", "usd_to_inr": f"{USD_TO_INR:.2f}",
                "note": note,
            })
    return cost, source, running


# ---- text-model step (used by the master prompt's analysis) ------------------
# USD per 1M tokens (input, output), from OpenAI's model page. Edit if prices change.
TEXT_RATES_PER_M = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
}


def log_text_cost(log_path, model, usage, note=""):
    """Log one text-model call. Returns (cost_inr, source, running_total_inr)."""
    u = _as_dict(usage)
    in_tok = int(u.get("input_tokens") or 0)
    out_tok = int(u.get("output_tokens") or 0)
    rate_in, rate_out = TEXT_RATES_PER_M.get(model, (0.0, 0.0))
    cost = (in_tok * rate_in + out_tok * rate_out) / 1_000_000 * USD_TO_INR
    source = "usage" if (in_tok or out_tok) and model in TEXT_RATES_PER_M else "unknown"
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        _archive_old_log(p)
        running = total_so_far(p) + cost
        new_file = not p.is_file()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "time": datetime.now().isoformat(timespec="seconds"), "step": "analysis",
                "model": model, "quality": "text", "size": "n/a",
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_inr": f"{cost:.4f}", "cost_source": source,
                "running_total_inr": f"{running:.4f}", "usd_to_inr": f"{USD_TO_INR:.2f}",
                "note": note,
            })
    return cost, source, running
