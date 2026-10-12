"""Hardcoded prompts for the preset mode. Edit these freely."""
import re
from pathlib import Path

COMMON_RULES = (
    "Keep the product from the uploaded image exactly as it is: same shape, "
    "colors, logo, labels and proportions. Do not alter, redraw or replace the "
    "product. Photorealistic, high quality, sharp focus."
)

PRESETS = {
    "studio": (
        "Create a professional studio product photo of the product. Clean "
        "seamless neutral background, soft diffused lighting, gentle natural "
        "shadow under the product, centered composition, e-commerce catalog style."
    ),
    "lifestyle": (
        "Place the product in a realistic lifestyle setting where it would "
        "naturally be used, such as a stylish home, desk or outdoor scene. "
        "Warm natural light, shallow depth of field, authentic and inviting mood."
    ),
    "ai_model": (
        "Show the product being worn, held or used by a realistic professional "
        "model in a polished fashion-editorial photo. Natural pose, flattering "
        "light, product clearly visible and the hero of the shot."
    ),
    "festival": (
        "Create a festive celebration-themed product photo. Warm glowing "
        "lights, decorative festive elements such as diyas, flowers and "
        "ornaments around the product, rich vibrant colors, joyful mood, "
        "product kept in clear focus at the center."
    ),
    "offers": (
        "Create a bold promotional sale-style product image. Eye-catching "
        "vibrant background with dynamic shapes, space left for offer text, "
        "product large and prominent, energetic advertising look. "
        "Do not add any text or numbers."
    ),
}

STYLES = list(PRESETS.keys())


def build_preset_prompt(style: str) -> str:
    return f"{PRESETS[style]} {COMMON_RULES}"


def build_custom_prompt(user_prompt: str) -> str:
    return f"{user_prompt.strip()} {COMMON_RULES}"


# ---------------------------------------------------------------------------
# MASTER PROMPT  (one uploaded image -> 3 finished ads: AI model, lifestyle, studio)
#
# The prompt text lives, word for word, in master_prompt.txt. It is loaded here
# and sent to the AI exactly as written; nothing in it is edited or added to.
# To change it, edit master_prompt.txt (not this file).
# ---------------------------------------------------------------------------
MASTER_PROMPT = (Path(__file__).parent / "master_prompt.txt").read_text(encoding="utf-8")

# Separate system-level note sent next to (never inside) the master prompt.
# The text model can only write; the app makes the images afterwards, so this tells it
# to stop after STEP 12 and print the winners under predictable headings.
# Set to "" to send the master prompt completely on its own.
MASTER_STAGE_NOTE = (
    "Image generation is performed by a separate system AFTER your reply, so you cannot "
    "and need not generate images yourself. Complete STEPS 1 to 12. In STEP 12, print the "
    "three final prompts under the exact headings 'WINNER 1', 'WINNER 2' and 'WINNER 3' "
    "(AI model, lifestyle, studio, in that order). Stop after WINNER 3; do not attempt STEP 13 or later."
)

# Winner number -> style name, in the order the master prompt defines them.
MASTER_STYLES = {1: "ai_model", 2: "lifestyle", 3: "studio"}

_WINNER_RE = re.compile(r"^[#*>\s]*WINNER\s*([123])\b[^\n]*\n", re.IGNORECASE | re.MULTILINE)
_STOP_RE = re.compile(
    r"^[#*>\s]*(?:STEP\s*1[1-9]\b|IMAGE\s*[123]\b|1[3-9]\.\s|FINAL OUTPUT ORDER)",
    re.IGNORECASE | re.MULTILINE,
)


def extract_winner_prompts(text: str) -> dict:
    """Pull the three winning prompts out of the text model's reply.

    Returns {"ai_model": "...", "lifestyle": "...", "studio": "..."}.
    Raises ValueError if a winner can't be found.
    """
    last = {}
    for m in _WINNER_RE.finditer(text):      # keep the LAST heading per number (the final list)
        last[int(m.group(1))] = m
    missing = [n for n in MASTER_STYLES if n not in last]
    if missing:
        raise ValueError(f"WINNER {missing} heading(s) not found in the model's reply")
    ordered = sorted(last.items(), key=lambda kv: kv[1].start())
    out = {}
    for i, (n, m) in enumerate(ordered):
        end = ordered[i + 1][1].start() if i + 1 < len(ordered) else len(text)
        chunk = text[m.end():end]
        stop = _STOP_RE.search(chunk)
        if stop:
            chunk = chunk[: stop.start()]
        chunk = chunk.strip().strip("-—_⸻* \n").strip()   # trailing --- or ⸻ rules
        if len(chunk) < 200:
            raise ValueError(f"WINNER {n} prompt looks too short ({len(chunk)} characters)")
        out[MASTER_STYLES[n]] = chunk
    return out
