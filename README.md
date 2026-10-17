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

Start the server as usual and open `http://localhost:8000/`. One page, three tabs:

- **New run**: pick models (flare, sunburst, or both), qualities (low, medium, high, one or several), a size,
  a one-line prompt and the product image. It shows a rough price before you start, then makes every
  model x quality combination (for example 2 models x 3 qualities = 6 images) in one numbered folder.
  Images appear as they finish, with the real cost under each. Buttons: open the shareable comparison page,
  download the whole run as a zip, copy the comparison link.
- **History**: every run with its prompt, images, cost and links.
- **Cost log**: `cost_log.csv` as a table, newest first, with a download link (`/cost.csv`).

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
  an unreadable reply is still charged, and it shows in the cost log.


## Review tab and Excel

The **Review** tab shows one card per generated image: run id, date, input image, initial prompt, text model and effort
(or Direct), the analysis, the prompt given to the image model, and the output image. The reviewer types a name once,
then clicks Yes / No (click again to clear) for: satisfied with the output, correct object identification, correct image
separation, happy with the background, product unchanged, lighting and shadows look natural, usable as an ad without edits,
plus an optional comment. Answers save immediately into `reviews.json` (next to `cost_log.csv`, not in git).
Several people can review the same image; each gets their own row.

`/review.xlsx` downloads it all: one row per image per reviewer, with pictures (or `?pictures=0` for links only),
`?run=<folder>` for one run, `?reviewed_only=1` to skip images nobody reviewed, plus a Summary sheet with Yes/No counts.
Links in the sheet use the address you opened the site on (or set `PUBLIC_URL=https://your-domain` in `.env`).
Needs `xlsxwriter` (in requirements.txt).
