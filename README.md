# Image Generation Backend (OpenAI)

Two endpoints. Both take an uploaded image and return a PNG.

| Endpoint | You send | Prompt comes from |
|---|---|---|
| `POST /generate/custom` | `image` file + `prompt` (one line) | You |
| `POST /generate/preset` | `image` file + `style` | Hardcoded in `prompts.py` |

`style` is one of: `studio`, `lifestyle`, `ai_model`, `festival`, `offers`.

## Setup

```bash
cd image-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your real OPENAI_API_KEY in .env
uvicorn main:app --reload
```

## Test

Easiest: open http://localhost:8000/docs, expand an endpoint, click
"Try it out", choose a file, and Execute.

Or with curl:

```bash
# Custom prompt
curl -X POST http://localhost:8000/generate/custom \
  -F "image=@product.png" \
  -F "prompt=on a wooden table with morning sunlight" \
  -o out.png

# Preset
curl -X POST http://localhost:8000/generate/preset \
  -F "image=@product.png" \
  -F "style=studio" \
  -o out.png
```

## Notes

- Change the prompts in `prompts.py`. `COMMON_RULES` is appended to every
  prompt to keep the product unchanged.
- `model` and `quality` can be sent with each request (see below). `.env` only sets the
  defaults used when a request leaves them out.
- Image edits are billed per image; `low` quality is cheapest for testing.
- Max upload is 10 MB (PNG, JPEG or WEBP).

## Testing models and tracking cost

`compare_models.py` makes ONE image by default (flare, quality high):

```bash
python compare_models.py product.png --style studio
python compare_models.py product.png --style studio --model gpt-image-2.5-sunburst --quality xhigh
python compare_models.py product.png --style studio --all --dry-run   # plan + estimate, no spend
python compare_models.py product.png --style studio --all             # all 10 (asks first)
```

- Images are saved as `comparisons/<time>/<model>__<quality>.png`, with an `index.html` to view them.
- Every call's cost is computed from the token usage OpenAI returns, printed, and appended to
  `comparisons/cost_log.csv` with a running total. If no usage is returned it falls back to a rough
  per-quality estimate and marks the row `estimate`.
- The backend (`main.py`) logs to `cost_log.csv` too. Each response carries an `X-Cost-INR` header,
  and `GET /cost` returns the total spent.
- Rates live in `costing.py`. Your OpenAI Usage page is the real source of truth.

## Choosing model and quality per request

Both endpoints accept optional form fields:

| Field | Values | If omitted |
|---|---|---|
| `model` | `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst`, `gpt-image-1` | `OPENAI_IMAGE_MODEL` |
| `quality` | `auto`, `low`, `medium`, `high`, `xhigh`, `max` | `OPENAI_IMAGE_QUALITY` |
| `size` | `auto` or WxH (multiples of 16, max edge 3840, ratio at most 3:1) | `1024x1024` |

`xhigh` and `max` work only on the 2.5 models. Invalid values are rejected before any
OpenAI call, so they cost nothing.

```bash
curl -X POST http://localhost:8000/generate/preset \
  -F "image=@product.png" -F "style=studio" \
  -F "model=gpt-image-2.5-sunburst" -F "quality=xhigh" \
  -D headers.txt -o out.png
```

The response headers `X-Model`, `X-Quality` and `X-Cost-INR` tell you what ran and what it cost.
`GET /health` lists the allowed models and qualities and the current defaults.
To allow another model, add it to the `Model` enum in `main.py`.

## Where images are saved

Every request gets its own numbered folder inside `outputs/`, named
`<number>_<model>_<quality>`, for example `outputs/1_gpt2.5flare_low/`:

```
outputs/
  1_gpt2.5flare_low/
    input.png          the image you uploaded
    ai_model.png       the generated ads (master mode: ai_model, lifestyle, studio)
    lifestyle.png
    studio.png
    analysis.md        the AI's written analysis (master mode only)
    comparison.html    input vs. every generated image
  2_gpt2.5sunburst_high/
    ...
```

`comparison.html` has the images embedded, so that single file can be emailed or shared and opens
in any browser (click an image to enlarge it). While the server runs you can also open it at
`http://localhost:8000/outputs/<folder>/comparison.html`.

The numbers keep counting up from the highest folder already in `outputs/`. The folder name is also
written in the `note` column of `cost_log.csv`, so each image can be matched to its cost.
`compare_models.py` still saves its images under `comparisons/<time>/`.

`outputs/index.html` is rebuilt after every run. It lists all runs (newest first) with the input and every
generated image, each run's cost, and a link to that run's own `comparison.html`. Open it from inside the
`outputs` folder, or at `http://localhost:8000/outputs/index.html`.

## Master prompt: one image in, three ads out

`POST /generate/master` takes one product image and returns three separate ads: AI model,
lifestyle and studio. The colleague's prompt is stored word for word in `master_prompt.txt`
and loaded by `prompts.py`; it is never edited by the code.

How it works:
1. A text model (`OPENAI_TEXT_MODEL`, default `gpt-6.1-sol`) reads your image and the master
   prompt, and writes three winning prompts (STEPS 1 to 12).
2. The app makes three separate image edits, one per winning prompt, using your image as the
   product reference.

Everything is saved in one numbered folder (see above). The response is JSON with the folder name,
a link to `comparison.html`, the three image files, the three winning prompts, and the cost in rupees. Optional fields: `model`, `quality`, `text_model`, `size`.
The prompt asks for 4:5 feed ads, so master mode defaults to `size=1088x1360` (exactly 4:5, valid for the API) and saves the images at that size.
One request costs one text call plus three image calls, and every part is logged to `cost_log.csv`.

## Web app (for your team)

Start the server as usual and open `http://localhost:8000/`. One page, four tabs (New run, Batch, Batch review, Scene library):

- **New run**: pick models (flare, sunburst, or both), qualities (low, medium, high, one or several), a size,
  a one-line prompt and the product image. It shows a rough price before you start, then makes every
  model x quality combination (for example 2 models x 3 qualities = 6 images) in one numbered folder.
  Images appear as they finish, with the real cost under each. Buttons: open the shareable comparison page,
  download the whole run as a zip, copy the comparison link.

Setting a password: add `APP_PASSWORD=something` to `.env`. The browser then asks for it (any user name).
Do this before putting the app on a public server, because every run spends your OpenAI credits.

### Running on EC2

```bash
# on the server (Ubuntu example), after copying the folder there:
cd image-backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY and APP_PASSWORD
uvicorn main:app --host 0.0.0.0 --port 8000
```

- Open port 8000 in the instance's security group (ideally only to your team's IPs).
- To keep it running after you log out, use a systemd service or `tmux`.
- A plain `http://` address sends the password unencrypted. Put nginx with HTTPS in front
  (set the proxy read timeout to 300 s), or only allow trusted IPs.
- A run keeps going in the background, so a long run does not time out the browser.
- `outputs/` and `cost_log.csv` live on the server's disk. Back them up or attach a volume if you need them long term.

### Text-model analysis step (optional)

Before the image model runs, the page can send your photo and prompt to a text model first.
The text model writes an analysis and a detailed image prompt, and that prompt goes to the image model:

```
image + prompt -> text model + effort -> analysis + written prompt -> image model + quality -> image
```

- Tick any of `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-luna` and any effort levels (`none` works with Luna only).
  Every text model x effort is its own branch, and each branch's prompt is sent to every image model x quality you tick.
- "Direct" sends your own prompt as it is (plus the standard preservation rules), so you can compare it with
  the AI-written prompts. If no text model is ticked, the page works as before.
- Images in one run = branches x image models x qualities. The cap is 15 (`WEB_MAX_IMAGES` in `.env`).
- The text model follows the instructions in `optimizer_prompt.txt`. Edit that file to change what it does,
  then restart the app.
- The comparison page shows the input, your prompt, and for each branch the model's analysis, the prompt it wrote,
  and its images, followed by a cost table with every text and image step.
- If a text step fails, its images are skipped and the other branches carry on. A text call that ran but gave
  an unreadable reply is still charged (costs are still written to `cost_log.csv` on the server).


## Nano Banana (Google)

`gemini-nano-banana-2.1` is in the image-model list next to flare and sunburst. It needs a Google key:
put `GEMINI_API_KEY=...` in `.env` on the server (key from https://aistudio.google.com/apikey, billing on), run
`pip install -r requirements.txt` (adds `google-genai`) and restart. Without the key the box is greyed out.
Nano Banana has no low/medium/high quality, so the page maps them to output size: low = 1K, medium = 2K, high = 4K
(price per image about $0.034 / $0.050 / $0.113). The picture shape follows the size you choose (1088x1360 -> 4:5).
Cost is computed from the token usage Google returns (input $1.50 and output incl. thinking $30 per 1M tokens).
Optional `NANO_THINKING=minimal|medium|high`. All Nano Banana images carry Google's SynthID watermark.


## Muse Image (Meta)

`muse-image-1.0` is in the image-model list. It uses Meta's Model API (OpenAI-compatible, `https://api.meta.ai/v1`).
Put `MODEL_API_KEY=...` in `.env` on the server (key from https://dev.meta.ai/ -> API keys) and restart; without it the box is
greyed out. Meta charges a flat $0.01 per image and Muse has no quality levels, so it always makes one image per branch
(labelled "standard") whatever qualities you tick. The size you pick only sets the aspect ratio; Meta decides the pixels.

## Batch tab and its own review sheet

Two extra tabs in the web page:

* **Batch**: the scene flow (see "Scene flow" below). The old "one prompt for many images" form was removed from the page;
  its server code is still in `batch.py` (`POST /api/batch`) but nothing in the web page uses it.
* **Batch review**: a separate review sheet (same 7 Yes/No questions + comment, Save / Edit review) saved in
  `batch_reviews.json`, with its own Excel download (`/batch_review.xlsx`). Each image shows its own cost (and the Excel has cost columns).

Limits: `WEB_BATCH_MAX` (default 100 images per batch), `WEB_BATCH_WORKERS` (default 3). Code: `batch.py`.

## Gemini and Meta text models

The analysis step (text model) can now also use **Gemini 4 Argon** and **Gemini 3.8 Flash** (Google, `GEMINI_API_KEY`; effort = thinking level low / medium / high)
and **Muse Spark 1.3** / **Muse Wonder 1.0** (Meta, `MODEL_API_KEY`; effort minimal to max). They appear in the "Analysis step" list and are greyed out
until the key is in the server's `.env`. Costs use the published prices (Argon $2 / $10 per 1M tokens until 30 Nov 2026, then $4 / $20; Flash $0.75 / $3.75 until 31 Dec 2026, then $1.50 / $7.50,
applied automatically; Spark $1.25 / $4.25; Wonder $4 / $20); thinking tokens are counted as output. Prices live in `costing.py` (`TEXT_RATES_PER_M`).

## Scene flow (Batch tab) and the Scene library

Run in three steps, one at a time (or "Run all steps"). Every uploaded photo is its **own product**; each step shows its result before you start the next.

1. **Identify**: Prompt A1 + the known categories + a text model and effort -> category and subcategory (an existing name is reused whenever the
   product fits; otherwise a new one is created), and how many scene descriptions already exist for that subcategory. You can change the category
   or subcategory by hand (this clears that product's scene and image).
2. **Scene**: Prompt A2 + the product + the earlier scenes of that subcategory + a text model and effort -> a NEW structured scene that must differ
   from the earlier ones. Prompt B is not used here. Photos in the same subcategory are done one after the other, so each sees the scene written for the one before.
Every run of step 2 (a **scene run**) and of step 3 (an **image run**) is kept for each product, numbered, with its model, effort, prompt, time and its own cost. Running step 2 again with another model or prompt adds a scene run instead of replacing the old one. Step 3 uses the scene run marked "used for step 3" (the newest by default; press *Use for step 3* on another one, or *Make an image from this scene*). Only that selected scene counts as "already used" for later products of the subcategory. The runs show in the Step 2 and Step 3 cards, in History (*Runs & costs*), in Batch review (one card per image, with its own cost chain) and in the Excel file (columns for the runs, plus an *Iterations* sheet listing every identify, scene and image run with its cost).
3. **Image**: Prompt C = your Prompt B with the scene description put where you wrote `{SCENE}` (if Prompt B has no `{SCENE}`, the scene comes first and Prompt B is added under it as "Additional instructions:"); the result is shown before you run -> gpt-image-2.5-sunburst, low.

* Each step can be re-run for one product (its later steps are cleared). A replaced scene no longer counts as "used" in the library.
* Prompts A1 and A2 are in `scene_classify_prompt.txt` and `scene_write_prompt.txt` and shown in editable boxes (Reset to default). The code adds the data blocks
  (known categories, earlier scenes, required JSON format) after them, so if the reply fields change, `sceneflow.py` (`SCENE_FORMAT`, `scene_library.SCENE_FIELDS`) must change too.
* Scenes are saved in `scene_library.json` (category > subcategory > scenes with timestamps; status pending / done / failed / replaced). Back this file up.
* Results stay saved between steps: close the page and open the run again from the Batch list. Finished images appear in Batch review (with their cost) and its Excel.
* Code: `sceneflow.py` (steps and API), `scene_library.py` (storage and name matching). Limit: `WEB_BATCH_MAX` photos per run (default 100).

Note: the page has five tabs: New run, Batch, Batch review, Scene library and History. History lists every single run and batch run, newest first, with a filter, a search box and an Open button for each (`GET /api/history`). The old Review and Cost log tabs were removed.
