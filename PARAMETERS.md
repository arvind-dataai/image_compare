# GPT Image 2.5: parameters (image edits)

Compiled from OpenAI's model pages and third-party docs (fal.ai, LLM Gateway) in Oct 2026.
Not every value is on OpenAI's own pages, so verify in your account before relying on one.

## Models
| Model ID | Meant for |
|---|---|
| `gpt-image-2.5-flare` | Fast, high-quality everyday generation (fastest) |
| `gpt-image-2.5-sunburst` | Most capable; precise edits and fine detail (slower) |

Dated snapshots (`...-2026-09-08`) are pinned versions for reproducible results.

## Parameters
| Parameter | Values | Meaning |
|---|---|---|
| `model` | see above | Which model runs |
| `image` | PNG / JPEG / WEBP | Input image(s) to edit |
| `prompt` | text | What to do |
| `quality` | `auto`, `low`, `medium`, `high`, `xhigh`, `max` | More thinking/detail per level: slower and costlier. Default is high per fal.ai |
| `size` | `auto`, or `WxH` | Each side a multiple of 16; max side 3840; total pixels 655,360 to 8,294,400; aspect ratio at most 3:1. e.g. `1024x1024`, `1536x1024`, `1024x1536` |
| `output_format` | `png`, `jpeg`, `webp` | File type returned |
| `output_compression` | 0 to 100 | JPEG/WEBP only; lower is smaller file, lower quality |
| `background` | `auto`, `opaque`, `transparent` | Transparent needs PNG or WEBP |
| `moderation` | `auto`, `low` | Strictness of content filtering |
| `n` | 1 to 10 | Number of images per request |
| `input_fidelity` | not documented for 2.5 | Don't send it unless you confirm it is accepted |

## Approx. cost per image at 1024x1024 (same for both models; from fal.ai, an estimate)
| Quality | Price |
|---|---|
| low | $0.006 |
| medium | $0.013 |
| high | $0.053 |
| xhigh | $0.094 |
| max | $0.211 |

Edits also bill the input image tokens, so real cost is a bit higher. Check your OpenAI Usage page.

## Rate limits (per OpenAI model pages)
Build tier: 20 images per minute. Free tier: not supported. May require API Organization Verification.
