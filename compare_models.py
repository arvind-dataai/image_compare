"""
Test image models on ONE input image + ONE prompt. By default this makes a
SINGLE image (one model, one quality). Every call's cost is printed and added
to a running log (comparisons/cost_log.csv).

Examples
--------
  # one image (default: gpt-image-2.5-flare, quality high)
  python compare_models.py product.png --style studio

  # pick the model / quality yourself
  python compare_models.py product.png --style studio --model gpt-image-2.5-sunburst --quality xhigh

  # a few specific combos
  python compare_models.py product.png --prompt "on a marble table" --model gpt-image-2.5-flare --quality low medium high

  # the full 2 models x 5 qualities grid (10 images; asks before spending)
  python compare_models.py product.png --style studio --all

  # show the plan and estimated cost without calling the API
  python compare_models.py product.png --style studio --all --dry-run

Output goes to comparisons/<timestamp>/ :
  <model>__<quality>.png   one image per run, named by model + quality
  results.csv, index.html  summary and side-by-side page for that run
And comparisons/cost_log.csv keeps a running total across ALL runs.
"""
import argparse
import base64
import csv
import html
import mimetypes
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from costing import USD_TO_INR, estimate_inr, log_cost, total_so_far
from prompts import STYLES, build_custom_prompt, build_preset_prompt

load_dotenv()

ALL_MODELS = ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
ALL_QUALITIES = ["low", "medium", "high", "xhigh", "max"]
DEFAULT_MODEL = "gpt-image-2.5-flare"
DEFAULT_QUALITY = "high"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="Path to the input image (PNG/JPEG/WEBP)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="One-line custom prompt")
    g.add_argument("--style", choices=STYLES, help="Use a hardcoded preset prompt")
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], dest="models")
    p.add_argument("--quality", nargs="+", default=[DEFAULT_QUALITY], dest="qualities")
    p.add_argument("--all", action="store_true", help="Run every model x every quality (10 images)")
    p.add_argument("--size", default="1024x1024", help='e.g. 1024x1024, 1536x1024, 1024x1536 or "auto"')
    p.add_argument("--format", default="png", choices=["png", "jpeg", "webp"], dest="fmt")
    p.add_argument("--out", default="comparisons", help="Base output folder")
    p.add_argument("--input-fidelity", choices=["low", "high"], default=None,
                   help="Only send if the model supports it (not documented for 2.5 models)")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and estimated cost, then exit")
    p.add_argument("--yes", action="store_true", help="Don't ask for confirmation when running more than one image")
    return p.parse_args()


def main():
    a = parse_args()
    img_path = Path(a.image)
    if not img_path.is_file():
        sys.exit(f"Input image not found: {img_path}")

    models = ALL_MODELS if a.all else a.models
    qualities = ALL_QUALITIES if a.all else a.qualities
    prompt = build_preset_prompt(a.style) if a.style else build_custom_prompt(a.prompt)
    runs = [(m, q) for m in models for q in qualities]
    est = sum(estimate_inr(q) for _, q in runs)
    log_path = Path(a.out) / "cost_log.csv"

    print(f"Input : {img_path}")
    print(f"Prompt: {prompt}")
    print(f"Size  : {a.size}   Format: {a.fmt}")
    print(f"Images to generate: {len(runs)}")
    for m, q in runs:
        print(f"  - {m} / {q}  (~₹{estimate_inr(q):.2f} est.)")
    print(f"Estimated cost: ~₹{est:.2f}  (rough, 1024x1024, at $1 = ₹{USD_TO_INR:.2f}; actual is computed from tokens after each call)")
    print(f"Spent so far (cost_log.csv): ₹{total_so_far(log_path):.2f}")
    if a.dry_run:
        print("\nDry run only. Nothing was sent.")
        return
    if len(runs) > 1 and not a.yes:
        if input(f"\nGenerate {len(runs)} images? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled.")

    out_dir = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"input{img_path.suffix}").write_bytes(img_path.read_bytes())
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    client = OpenAI()
    mime = mimetypes.guess_type(img_path.name)[0] or "image/png"
    image_bytes = img_path.read_bytes()
    rows = []
    run_total = 0.0
    print()

    for i, (model, quality) in enumerate(runs, 1):
        fname = f"{model}__{quality}.{a.fmt}"
        print(f"[{i}/{len(runs)}] {model} / {quality} ...", end=" ", flush=True)
        kwargs = dict(
            model=model,
            image=(img_path.name, image_bytes, mime),
            prompt=prompt,
            size=a.size,
            quality=quality,
            output_format=a.fmt,
        )
        if a.input_fidelity:
            kwargs["input_fidelity"] = a.input_fidelity
        row = {"model": model, "quality": quality, "file": "", "seconds": "", "status": "",
               "cost_inr": "", "cost_source": "", "error": ""}
        t0 = time.time()
        try:
            result = client.images.edit(**kwargs)
            row["seconds"] = round(time.time() - t0, 1)
            (out_dir / fname).write_bytes(base64.b64decode(result.data[0].b64_json))
            cost, source, running = log_cost(
                log_path, model, quality, a.size, getattr(result, "usage", None), note=img_path.name
            )
            run_total += cost
            row.update(file=fname, status="ok", cost_inr=f"{cost:.4f}", cost_source=source)
            tag = "" if source == "usage" else " (estimate, no usage returned)"
            print(f"ok  {row['seconds']}s  cost ₹{cost:.2f}{tag}")
        except OpenAIError as e:
            row["seconds"] = round(time.time() - t0, 1)
            row.update(status="error", error=str(e)[:300])
            print("ERROR:", str(e)[:120])
        rows.append(row)

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cards = []
    for r in rows:
        if r["status"] == "ok":
            body = f'<img src="{html.escape(r["file"])}" loading="lazy">'
            meta = f'quality: {html.escape(r["quality"])} &middot; {r["seconds"]}s &middot; &#8377;{r["cost_inr"]}'
        else:
            body = f'<p class="err">{html.escape(r["error"])}</p>'
            meta = f'quality: {html.escape(r["quality"])}'
        cards.append(f'<figure><figcaption><b>{html.escape(r["model"])}</b><br>{meta}</figcaption>{body}</figure>')
    page = f"""<!doctype html><meta charset="utf-8"><title>Model comparison</title>
<style>
body{{font:14px system-ui;margin:20px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}}
img{{width:100%;border-radius:4px}} .err{{color:#b00020;word-break:break-word}}
</style>
<h2>Model comparison</h2>
<p><b>Prompt:</b> {html.escape(prompt)}</p>
<p><b>This run:</b> &#8377;{run_total:.2f}</p>
<p><b>Input:</b><br><img src="input{img_path.suffix}" style="max-width:260px"></p>
<div class="grid">{''.join(cards)}</div>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nDone: {ok}/{len(rows)} succeeded.")
    print(f"This run cost : ₹{run_total:.2f}")
    print(f"Total spent   : ₹{total_so_far(log_path):.2f}   (log: {log_path})")
    print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
