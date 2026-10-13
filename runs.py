"""One output folder per request, plus a self-contained HTML page to compare results.

Folder name: <number>_<model>_<quality>, e.g. outputs/1_gpt2.5flare_low/
  input.png        the image you uploaded
  ai_model.png ... the generated image(s)
  analysis.md      the AI's written analysis (master mode only)
  comparison.html  input vs. every generated image, shareable as a single file
"""
import base64
import html
import json
import re
import threading
from datetime import datetime
from pathlib import Path

from costing import USD_TO_INR

_lock = threading.Lock()
_NUM = re.compile(r"^(\d+)_")
_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def short_model(model: str) -> str:
    """gpt-image-2.5-flare -> gpt2.5flare"""
    return model.replace("gpt-image-", "gpt").replace("-", "")


def ext_for(mime: str) -> str:
    return _EXT.get(mime, ".png")


def new_run_dir(output_dir: Path, model: str, quality: str) -> Path:
    """Create the next numbered folder, e.g. 3_gpt2.5flare_low."""
    output_dir.mkdir(exist_ok=True)
    with _lock:
        nums = [int(m.group(1)) for p in output_dir.iterdir() if p.is_dir() and (m := _NUM.match(p.name))]
        run_dir = output_dir / f"{max(nums, default=0) + 1}_{short_model(model)}_{quality}"
        run_dir.mkdir()
    return run_dir


def _data_uri(path: Path) -> str:
    mime = _MIME.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


_CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#1c1c1a;--mute:#6b6b66;--line:#e2e2dc;--accent:#2a5bd7;--err:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#202020;--ink:#ececea;--mute:#a0a09a;--line:#333;--accent:#8ab4ff;--err:#f2b8b5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}
.meta{color:var(--mute);margin:0 0 20px;font-size:14px}
.meta b{color:var(--ink);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card.input{border:2px solid var(--accent)}
.card img{display:block;width:100%;height:auto;cursor:zoom-in;background:#0001}
.cap{padding:10px 12px}
.cap .t{font-weight:600}
.cap .s{color:var(--mute);font-size:13px}
.costs{margin:0 0 20px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.costs h2{font-size:16px;margin:0 0 8px}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td:first-child{white-space:normal}
.n{text-align:right;font-variant-numeric:tabular-nums}
.tot td{font-weight:700;border-bottom:0}
.sec td{background:#f1f3f7;font-weight:700;font-size:.85em;text-transform:uppercase;letter-spacing:.04em}
.sub td{font-weight:600;background:#fafbfc}
.note{color:var(--mute);font-size:12.5px;margin:8px 0 0}
.err{padding:16px 12px;color:var(--err);word-break:break-word;font-size:14px}
details{border-top:1px solid var(--line);padding:8px 12px;font-size:13px}
summary{cursor:pointer;color:var(--accent)}
pre{white-space:pre-wrap;word-break:break-word;margin:8px 0 0;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
.analysis{margin-top:20px;background:var(--card);border:1px solid var(--line);border-radius:10px}
.analysis details{border-top:0;padding:12px 16px}
#lb{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out;padding:16px}
#lb img{max-width:100%;max-height:100%;border-radius:6px}
"""

_JS = """
const lb=document.getElementById('lb'),lbi=lb.querySelector('img');
document.querySelectorAll('.card img').forEach(i=>i.addEventListener('click',()=>{lbi.src=i.src;lb.style.display='flex'}));
lb.addEventListener('click',()=>lb.style.display='none');
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.style.display='none'});
"""


def split_costs(costs) -> tuple:
    """(analysis_total, image_total) in rupees; a step is 'analysis' if its label starts with Analysis."""
    ana = sum(c["cost_inr"] for c in costs or [] if c["part"].startswith("Analysis"))
    img = sum(c["cost_inr"] for c in costs or [] if not c["part"].startswith("Analysis"))
    return ana, img


def _cost_block(costs) -> str:
    """The 'What it cost' table: analysis steps and image steps listed apart, each with a subtotal."""
    if not costs:
        return ""
    e = html.escape

    def rows_for(items):
        return "".join(
            f'<tr><td>{e(c["part"])}</td><td>{e(c["model"])}</td>'
            f'<td class="n">{c["tokens_in"]:,}</td><td class="n">{c["tokens_out"]:,}</td>'
            f'<td class="n">\u20b9{c["cost_inr"]:.2f}</td><td>{e(c.get("source", ""))}</td></tr>'
            for c in items)

    def section(title, items):
        if not items:
            return ""
        sub = sum(c["cost_inr"] for c in items)
        return (f'<tr class="sec"><td colspan="6">{title}</td></tr>{rows_for(items)}'
                f'<tr class="sub"><td colspan="4">Subtotal: {title.lower()}</td>'
                f'<td class="n">\u20b9{sub:.2f}</td><td></td></tr>')

    ana = [c for c in costs if c["part"].startswith("Analysis")]
    img = [c for c in costs if not c["part"].startswith("Analysis")]
    total = sum(c["cost_inr"] for c in costs)
    return (
        '<div class="costs"><h2>What it cost to generate</h2>'
        f'<p><b>Analysis \u20b9{sum(c["cost_inr"] for c in ana):.2f} \u00b7 Images \u20b9{sum(c["cost_inr"] for c in img):.2f} '
        f'\u00b7 Total \u20b9{total:.2f}</b></p><div class="tw"><table>'
        '<tr><th>Step</th><th>Model</th><th class="n">Tokens in</th><th class="n">Tokens out</th>'
        f'<th class="n">Cost (\u20b9)</th><th>Source</th></tr>'
        f'{section("Analysis (text model)", ana)}{section("Image generation", img)}'
        f'<tr class="tot"><td colspan="4">Total (analysis + images)</td><td class="n">\u20b9{total:.2f}</td><td></td></tr>'
        f'</table></div><p class="note">Costs are computed from the token usage OpenAI returned, converted at '
        f'\u20b9{USD_TO_INR:.2f} per US dollar. "estimate" means OpenAI sent no usage data.</p></div>'
    )


def write_comparison(run_dir: Path, title: str, input_file: str, items: list, meta: dict, analysis: str = "",
                     costs: list | None = None, prompt: str = "", full_prompt: str = "") -> Path:
    """Write comparison.html (images embedded, so the single file can be emailed or shared).

    items: [{"label": str, "file": str|None, "cost_inr": float|None, "prompt": str, "error": str|None}]
    meta:  {"Model": ..., "Quality": ..., ...}  shown under the title.
    """
    e = html.escape
    cards = [
        f'<div class="card input"><img src="{_data_uri(run_dir / input_file)}" alt="Input image">'
        f'<div class="cap"><div class="t">Original (input)</div><div class="s">The image you uploaded</div></div></div>'
    ]
    for it in items:
        if it.get("file"):
            body = f'<img src="{_data_uri(run_dir / it["file"])}" alt="{e(it["label"])}">'
        else:
            body = f'<div class="err">Not generated: {e(str(it.get("error") or "unknown error"))}</div>'
        sub = []
        if it.get("cost_inr") is not None:
            sub.append(f'₹{it["cost_inr"]:.2f}')
        if it.get("file"):
            sub.append(e(it["file"]))
        prompt_html = (f'<details><summary>Prompt used</summary><pre>{e(it["prompt"])}</pre></details>'
                  if it.get("prompt") else "")
        cards.append(
            f'<div class="card">{body}<div class="cap"><div class="t">{e(it["label"])}</div>'
            f'<div class="s">{" &middot; ".join(sub)}</div></div>{prompt_html}</div>'
        )
    meta_line = " &nbsp;|&nbsp; ".join(
        f"{e(k)}: <b>{e(str(v))}</b>" for k, v in meta.items() if not (prompt and k == "Prompt"))
    prompt_block = ""
    if prompt:
        extra = (f'<p class="note">Full text sent to the model (your prompt plus the standard product-preservation rules):</p>'
                 f"<pre>{e(full_prompt)}</pre>") if full_prompt else ""
        prompt_block = (f'<div class="costs"><details><summary>Prompt used</summary>'
                        f'<pre>{e(prompt)}</pre>{extra}</details></div>')
    cost_block = _cost_block(costs)
    analysis_block = (
        f'<div class="analysis"><details><summary>Full AI analysis (steps 1 to 10)</summary>'
        f"<pre>{e(analysis)}</pre></details></div>" if analysis else ""
    )
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>'
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">"
        f"<h1>{e(title)}</h1>"
        f'<p class="meta">{meta_line} &nbsp;|&nbsp; {datetime.now():%d %b %Y, %H:%M}</p>'
        f'{prompt_block}{cost_block}<div class="grid">{"".join(cards)}</div>{analysis_block}</div>'
        f'<div id="lb"><img alt=""></div><script>{_JS}</script></body></html>'
    )
    out = run_dir / "comparison.html"
    out.write_text(page, encoding="utf-8")
    total = sum(c["cost_inr"] for c in costs) if costs else sum(i.get("cost_inr") or 0 for i in items)
    (run_dir / "run.json").write_text(json.dumps({
        "title": title, "input": input_file, "meta": meta, "total_inr": total,
        "items": [{"label": i["label"], "file": i.get("file"), "cost_inr": i.get("cost_inr")} for i in items],
        "time": f"{datetime.now():%d %b %Y, %H:%M}",
    }, ensure_ascii=False), encoding="utf-8")
    try:
        write_index(run_dir.parent)
    except Exception:
        pass   # the index is a convenience; never fail a run because of it
    return out


def write_index(output_dir: Path) -> Path:
    """outputs/index.html: every run on one page (newest first), linking to each run's own page.

    Images are linked by relative path, so open it from inside the outputs folder (or share the folder).
    """
    e = html.escape
    runs = []
    for p in output_dir.iterdir():
        mm = _NUM.match(p.name)
        if p.is_dir() and mm and (p / "run.json").is_file():
            try:
                runs.append((int(mm.group(1)), p, json.loads((p / "run.json").read_text(encoding="utf-8"))))
            except Exception:
                continue
    runs.sort(key=lambda t: -t[0])
    grand = sum(d.get("total_inr") or 0 for _, _, d in runs)
    sections = []
    for _, p, d in runs:
        n = e(p.name)
        thumbs = [f'<figure><img loading="lazy" src="{n}/{e(d["input"])}" alt="Input"><figcaption><b>Original (input)</b></figcaption></figure>']
        for it in d["items"]:
            if it.get("file"):
                cost = f' &middot; \u20b9{it["cost_inr"]:.2f}' if it.get("cost_inr") is not None else ""
                thumbs.append(f'<figure><img loading="lazy" src="{n}/{e(it["file"])}" alt="{e(it["label"])}">'
                              f'<figcaption><b>{e(it["label"])}</b>{cost}</figcaption></figure>')
            else:
                thumbs.append(f'<figure><div class="err">{e(it["label"])}: not generated</div></figure>')
        meta = " &nbsp;|&nbsp; ".join(f"{e(k)}: <b>{e(str(v))}</b>" for k, v in d["meta"].items() if k not in ("Cost", "Total cost"))
        sections.append(
            f'<section><h2>{n}</h2><p class="meta">{meta} &nbsp;|&nbsp; Run cost: <b>\u20b9{d["total_inr"]:.2f}</b>'
            f' &nbsp;|&nbsp; {e(d.get("time", ""))} &nbsp;|&nbsp; <a href="{n}/comparison.html">Open full comparison &rarr;</a></p>'
            f'<div class="grid">{"".join(thumbs)}</div></section>'
        )
    css = _CSS + (
        "section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 18px}"
        "section h2{font-size:17px;margin:0 0 2px}a{color:var(--accent)}"
        "figure{margin:0}figure img{display:block;width:100%;height:auto;border-radius:8px;border:1px solid var(--line)}"
        "figcaption{font-size:13px;color:var(--mute);padding:6px 2px}figcaption b{color:var(--ink)}"
        ".grid{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}"
    )
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>All runs</title>'
        f'<style>{css}</style></head><body><div class="wrap"><h1>All runs</h1>'
        f'<p class="meta">{len(runs)} run(s) &nbsp;|&nbsp; Total spent: <b>\u20b9{grand:.2f}</b></p>'
        f'{"".join(sections) or "<p>No runs yet.</p>"}</div></body></html>'
    )
    out = output_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


_FLOW_CSS = """
.top{display:grid;grid-template-columns:minmax(160px,260px) 1fr;gap:16px;align-items:start;margin:0 0 20px}
@media (max-width:640px){.top{grid-template-columns:1fr}}
.top .card{margin:0}
.ptxt{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.ptxt h2,.branch h2{font-size:16px;margin:0 0 6px}
.branch{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 20px}
.bh{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;margin:0 0 10px}
.bh .t{font-size:17px;font-weight:700}
.chip{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:999px;padding:1px 10px;color:var(--mute)}
.chip.ok{color:var(--accent);border-color:var(--accent)}
.box{border:1px solid var(--line);border-radius:8px;padding:6px 12px;margin:0 0 10px;background:var(--bg)}
.box details{border:0;padding:4px 0}
.box summary{font-weight:600}
.box pre{max-height:360px;overflow:auto;font-family:inherit;font-size:13.5px}
.box.prompt{border-color:var(--accent)}
.arrow{color:var(--mute);font-size:13px;margin:4px 0 10px}
.arrow:before{content:"\\2193  "}
"""


def write_flow_comparison(run_dir: Path, title: str, input_file: str, prompt: str, branches: list,
                          meta: dict, costs: list | None = None) -> Path:
    """comparison.html for a text-model -> image-model run, plus run.json for History.

    branches: [{label, kind ("text"|"direct"), text_model?, effort?, analysis?, final_prompt?, error?,
                text_cost_inr?, items: [{label, file, cost_inr, error}]}]
    """
    e = html.escape
    parts = []
    for b in branches:
        costs_txt = f'<span class="chip">analysis \u20b9{b["text_cost_inr"]:.2f}</span>' if b.get("text_cost_inr") is not None else ""
        if b["kind"] == "text":
            head = (f'<div class="bh"><span class="t">{e(b["text_model"])}</span>'
                    f'<span class="chip">effort: {e(b["effort"])}</span>{costs_txt}</div>')
            box = ""
            if b.get("analysis"):
                box += (f'<div class="box"><details open><summary>Analysis by {e(b["text_model"])} ({e(b["effort"])})</summary>'
                        f'<pre>{e(b["analysis"])}</pre></details></div>')
            if b.get("final_prompt"):
                box += (f'<div class="box prompt"><details open><summary>Prompt it wrote for the image model</summary>'
                        f'<pre>{e(b["final_prompt"])}</pre></details></div>')
            if b.get("error"):
                box += f'<div class="err">{e(b["error"])}</div>'
        else:
            head = '<div class="bh"><span class="t">Direct</span><span class="chip">your prompt, no text model</span></div>'
            box = (f'<div class="box prompt"><details><summary>Prompt sent to the image model (your prompt plus the standard '
                   f'preservation rules)</summary><pre>{e(b.get("final_prompt", ""))}</pre></details></div>')
        cards = []
        for it in b["items"]:
            if it.get("file"):
                body = f'<img src="{_data_uri(run_dir / it["file"])}" alt="{e(it["label"])}">'
            else:
                body = f'<div class="err">Not generated: {e(str(it.get("error") or "unknown error"))}</div>'
            sub = f'\u20b9{it["cost_inr"]:.2f}' if it.get("cost_inr") is not None else ""
            cards.append(f'<div class="card">{body}<div class="cap"><div class="t">{e(it["label"])}</div>'
                         f'<div class="s">{sub}</div></div></div>')
        parts.append(f'<div class="branch">{head}{box}<div class="arrow">sent to the image model(s)</div>'
                     f'<div class="grid">{"".join(cards)}</div></div>')

    meta_line = " &nbsp;|&nbsp; ".join(f"{e(k)}: <b>{e(str(v))}</b>" for k, v in meta.items() if k != "Prompt")
    top = (f'<div class="top"><div class="card input"><img src="{_data_uri(run_dir / input_file)}" alt="Input image">'
           f'<div class="cap"><div class="t">Original (input)</div></div></div>'
           f'<div class="ptxt"><h2>Your prompt</h2><pre style="font-family:inherit;font-size:14px">{e(prompt)}</pre>'
           f'<p class="note">Flow: image + prompt \u2192 text model (analysis, written prompt) \u2192 image model \u2192 image. '
           f'Each text model and effort below is its own branch.</p></div></div>')
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>'
        f"<style>{_CSS}{_FLOW_CSS}</style></head><body><div class=\"wrap\">"
        f"<h1>{e(title)}</h1>"
        f'<p class="meta">{meta_line} &nbsp;|&nbsp; {datetime.now():%d %b %Y, %H:%M}</p>'
        f'{top}{_cost_block(costs)}{"".join(parts)}</div>'
        f'<div id="lb"><img alt=""></div><script>{_JS}</script></body></html>'
    )
    (run_dir / "comparison.html").write_text(page, encoding="utf-8")

    only_direct = all(b["kind"] == "direct" for b in branches)
    flat = []
    for b in branches:
        for it in b["items"]:
            flat.append({"label": it["label"] if only_direct else f'{b["label"]} \u2192 {it["label"]}',
                         "file": it.get("file"), "cost_inr": it.get("cost_inr")})
    total = sum(c["cost_inr"] for c in costs) if costs else sum(i.get("cost_inr") or 0 for i in flat)
    keep = ("id", "kind", "label", "text_model", "effort", "analysis", "final_prompt", "error", "text_cost_inr", "status")
    (run_dir / "run.json").write_text(json.dumps({
        "title": title, "input": input_file, "meta": meta, "total_inr": total, "items": flat,
        "branches": [{**{k: b[k] for k in keep if k in b},
                      "items": [{k: it.get(k) for k in ("label", "model", "quality", "file", "cost_inr", "error", "status")}
                                for it in b["items"]]} for b in branches],
        "time": f"{datetime.now():%d %b %Y, %H:%M}",
    }, ensure_ascii=False), encoding="utf-8")
    try:
        write_index(run_dir.parent)
    except Exception:
        pass
    return run_dir / "comparison.html"
