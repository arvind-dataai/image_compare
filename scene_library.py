"""The scene library: category > subcategory > scenes (each with a timestamp), saved in scene_library.json.

Used by the batch "scene flow" (sceneflow.py). Pure storage + text helpers, no web code, no AI calls.

File layout (human readable, safe to open in any editor):

{
 "categories": {
   "Jewellery": {
     "subcategories": {
       "Bangles": {
         "scenes": [
           {"id": "a1b2c3d4", "created": "2026-10-23T15:40:12", "status": "done" | "pending" | "failed" | "replaced",
            "folder": "batch_3_...", "file": "out_001_....png", "input": "in_001_....jpg",
            "text_model": "...", "effort": "...", "prompt_b": "...",
            "scene": {...structured fields written by the text model...},
            "scene_description": "the rendered text that went into Prompt C",
            "summary": "one line used to compare with later scenes"}
         ]}}}}}
"""
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import main as core

LIB_FILE = Path(core.COST_LOG).parent / "scene_library.json"
_LOCK = threading.Lock()

# The fixed structure of one scene. key -> heading used when the scene is rendered as text for the image model.
SCENE_FIELDS = [
    ("concept_name", "CONCEPT"),
    ("setting_environment", "SETTING"),
    ("surface_and_background", "SURFACE AND BACKGROUND"),
    ("product_presentation", "PRODUCT PRESENTATION"),
    ("composition", "COMPOSITION"),
    ("camera_angle_and_framing", "CAMERA"),
    ("lighting", "LIGHTING"),
    ("colour_palette", "COLOUR PALETTE"),
    ("props", "PROPS"),
    ("mood_and_style", "MOOD AND STYLE"),
]
SCENE_KEYS = [k for k, _ in SCENE_FIELDS]
CLASSIFY_KEYS = ["category", "subcategory", "product_description"]


# ---------------------------------------------------------------- names
def key_of(name: str) -> str:
    """Spelling-proof key: 'Bangle', 'bangles ' and 'BANGLES' all give 'bangle'."""
    k = re.sub(r"[^a-z0-9]+", "", str(name).lower().replace("&", "and"))
    if len(k) > 3:
        if k.endswith("ies"):
            return k[:-3] + "y"
        if re.search(r"(ss|x|ch|sh|z)es$", k):
            return k[:-2]
        if k.endswith("s") and not k.endswith("ss"):
            return k[:-1]
    return k


def clean_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name)).strip(" .,-_/")
    return s[:60].title() if s and s == s.lower() else s[:60]


# ---------------------------------------------------------------- file
def _load() -> dict:
    try:
        d = json.loads(LIB_FILE.read_text(encoding="utf-8"))
        d.setdefault("categories", {})
        return d
    except Exception:
        return {"categories": {}}


def _save(d: dict):
    tmp = LIB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(LIB_FILE)


def taxonomy() -> dict:
    """{category: [subcategories]} as they exist now (also those with no scene yet)."""
    with _LOCK:
        d = _load()
    return {c: list(v.get("subcategories", {})) for c, v in d["categories"].items()}


def taxonomy_text() -> str:
    t = taxonomy()
    if not t:
        return "(the library is empty: no categories yet, so create the first category and subcategory)"
    return "\n".join(f"- {c}: " + (", ".join(s) if s else "(no subcategories yet)") for c, s in t.items())


def resolve(category: str, subcategory: str) -> tuple:
    """Fit the model's names onto the library. Returns (category, subcategory, new_category, new_subcategory).
    Existing spellings always win, so 'Bangle' lands on the existing 'Bangles'."""
    category, subcategory = clean_name(category), clean_name(subcategory)
    t = taxonomy()
    cat = next((c for c in t if key_of(c) == key_of(category)), None)
    if cat is None:
        return category, subcategory, True, True
    sub = next((s for s in t[cat] if key_of(s) == key_of(subcategory)), None)
    return cat, (sub or subcategory), False, sub is None


# ---------------------------------------------------------------- scenes
def summary_of(scene: dict) -> str:
    def cut(k, n=110):
        v = str(scene.get(k, "")).replace("\n", " ").strip()
        return v if len(v) <= n else v[: n - 1] + "…"
    return (f"concept: {cut('concept_name', 60)}; setting: {cut('setting_environment')}; surface/background: {cut('surface_and_background')}; "
            f"palette: {cut('colour_palette')}; lighting: {cut('lighting')}; camera: {cut('camera_angle_and_framing')}; props: {cut('props')}")


def previous(category: str, subcategory: str, limit: int = 40) -> list:
    """Scenes already used for this subcategory (newest last). Failed and replaced ones do not count as used."""
    with _LOCK:
        d = _load()
    sc = d["categories"].get(category, {}).get("subcategories", {}).get(subcategory, {}).get("scenes", [])
    return [s for s in sc if s.get("status") not in ("failed", "replaced")][-limit:]


def add_scene(category: str, subcategory: str, entry: dict) -> str:
    """Save a scene (status 'pending' until its image is made). Creates the category / subcategory if new."""
    entry = dict(entry, id=uuid.uuid4().hex[:8], created=datetime.now().isoformat(timespec="seconds"))
    entry.setdefault("status", "pending")
    with _LOCK:
        d = _load()
        cat = d["categories"].setdefault(category, {"subcategories": {}})
        sub = cat["subcategories"].setdefault(subcategory, {"scenes": []})
        sub["scenes"].append(entry)
        _save(d)
    return entry["id"]


def ensure(category: str, subcategory: str):
    """Register a (new) category / subcategory right away, so the next image of the same batch fits into it."""
    with _LOCK:
        d = _load()
        d["categories"].setdefault(category, {"subcategories": {}})["subcategories"].setdefault(subcategory, {"scenes": []})
        _save(d)


def update_scene(scene_id: str, **fields):
    with _LOCK:
        d = _load()
        for c in d["categories"].values():
            for s in c.get("subcategories", {}).values():
                for sc in s.get("scenes", []):
                    if sc.get("id") == scene_id:
                        sc.update(fields)
                        _save(d)
                        return True
    return False


def overview() -> list:
    """For the Scene library tab."""
    with _LOCK:
        d = _load()
    out = []
    for c, cv in d["categories"].items():
        subs = []
        for s, sv in cv.get("subcategories", {}).items():
            subs.append({"name": s, "scenes": [
                {"id": x.get("id"), "created": x.get("created", ""), "status": x.get("status", ""),
                 "concept": x.get("scene", {}).get("concept_name", ""), "summary": x.get("summary", ""),
                 "folder": x.get("folder", ""), "file": x.get("file", "")} for x in sv.get("scenes", [])]})
        out.append({"name": c, "subcategories": subs})
    return out


# ---------------------------------------------------------------- text helpers
def render_scene(category: str, subcategory: str, product_description: str, scene: dict) -> str:
    """The Scene half of Prompt C: the same labelled sections every time, filled with this scene's details."""
    def val(v):
        return "; ".join(str(x) for x in v) if isinstance(v, list) else str(v).strip()
    head = f"PRODUCT: {val(product_description)} ({category} > {subcategory})."
    head += " The attached photo shows the product."
    lines = [head]
    for k, label in SCENE_FIELDS:
        v = val(scene.get(k, ""))
        if v:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def build_prompt_c(scene_description: str, prompt_b: str) -> str:
    """Prompt C. If Prompt B contains the placeholder {SCENE}, the scene description goes exactly there.
    Without the placeholder, Prompt B is added under the scene description (the old behaviour)."""
    prompt_b = (prompt_b or "").strip()
    if re.search(r"\{\s*scene\s*\}", prompt_b, flags=re.I):
        return re.sub(r"\{\s*scene\s*\}", lambda m: scene_description, prompt_b, flags=re.I)
    return scene_description + (f"\n\nAdditional instructions:\n{prompt_b}" if prompt_b else "")


def parse_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (tolerates ```json fences and chatter around it)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("the reply contains no JSON object")
    return json.loads(t[i: j + 1])
