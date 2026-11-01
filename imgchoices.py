"""The image models you can pick in step 3 of R1-R4: id -> (model, quality). Cost estimates come from costing.estimate_inr."""
from costing import estimate_inr

CHOICES = {
    "sunburst-low": ("gpt-image-2.5-sunburst", "low", "Sunburst · low"),
    "flare-low": ("gpt-image-2.5-flare", "low", "Flare · low"),
    "nano": ("gemini-nano-banana-2.1", "low", "Nano Banana (1K)"),
    "muse": ("muse-image-1.0", "low", "Muse Image"),
}
DEFAULT = "sunburst-low"


def pick(cid: str):
    """(model, quality) for a choice id; an empty id means the default."""
    from fastapi import HTTPException
    c = CHOICES.get(cid or DEFAULT)
    if not c:
        raise HTTPException(400, "Choose the image model.")
    return c[0], c[1]


def for_page() -> list:
    return [{"id": k, "model": m, "quality": q, "label": lab, "est_inr_1024": round(estimate_inr(q, m), 4), "flat": not m.startswith("gpt-image")}
            for k, (m, q, lab) in CHOICES.items()]
