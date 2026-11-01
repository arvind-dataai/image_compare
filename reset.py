"""POST /api/reset : make the app completely fresh (History, runs, images, costs, groups, in-memory state).
Scene libraries are kept unless "libraries" is true. Refused while anything is still running."""
import shutil
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException

import batch
import main as core
import r1
import rflow
import webapp

router = APIRouter()
BASE = Path(core.COST_LOG).parent


def _busy() -> bool:
    if any(j.get("state") == "running" for j in list(webapp.JOBS.values()) + list(webapp.GROUPS.values()) + list(batch.BJOBS.values())):
        return True
    for jobs in (r1.JOBS, rflow.JOBS):
        for job in list(jobs.values()):
            if any(it.get("running") for it in job.get("items", [])):
                return True
    return False


@router.post("/api/reset")
def reset_all(confirm: str = Form(""), libraries: bool = Form(False)):
    if confirm.strip() != "RESET":
        raise HTTPException(400, 'Type RESET to confirm.')
    if _busy():
        raise HTTPException(409, "Something is still running. Wait until it finishes, then reset.")
    with r1.LOCK, rflow.LOCK:
        for p in core.OUTPUT_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.name != ".gitkeep":
                p.unlink(missing_ok=True)
        for p in list(BASE.glob("cost_log*.csv")) + [webapp.GROUPS_FILE]:
            p.unlink(missing_ok=True)
        if libraries:
            for name in ("scene_library.json", "r2_library.json", "r3_library.json", "r4_library.json"):
                (BASE / name).unlink(missing_ok=True)
        for d in (webapp.JOBS, webapp.GROUPS, batch.BJOBS, r1.JOBS, rflow.JOBS):
            d.clear()
    return {"ok": True, "libraries_deleted": bool(libraries)}
