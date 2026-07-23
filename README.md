# ComfyUI Booru Tagger

## Modification from [pythongosssss](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

1. Migrate to ComfyUI Node v3.
2. Separate model loading and inference, much faster running! (No longer need to load models for each image input).
3. **New model support**
    - [Pixai Tagger v0.9 (onnx model)](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)
    - [Camie Tagger v2](https://huggingface.co/Camais03/camie-tagger-v2)
    - [CL Tagger v1 (1.00 / 1.01 / 1.02)](https://huggingface.co/cella110n/cl_tagger)
    - [CL Tagger v2 (2.00 / 2.01a)](https://huggingface.co/cella110n/cl_tagger_v2)
    - [AnimeTimm dbv4-full](https://huggingface.co/animetimm) — timm-based taggers with 12,476 tags (caformer, eva02, swinv2, and ConvNeXtV2 Huge)
4. **Multiple output fields** — `tags` (combined), `general_tags`, `rating`, `character_tags`

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension allowing the interrogation of booru tags from images.

For AnimeTimm models, **use_best_threshold** defaults to enabled and applies each tag's model-provided `best_threshold` as a minimum. Disable it to use only the node's `threshold` and `character_threshold` inputs, which is useful when you prefer higher recall. The defaults are the model author's category-level recommendations: eva02 (`0.39` / `0.61`), caformer (`0.39` / `0.47`), swinv2 (`0.41` / `0.59`), and ConvNeXtV2 Huge (`0.38` / `0.51`) for general / character tags.

## Outputs

| Output | Description |
|---|---|
| `tags` | Combined character + general tags |
| `general_tags` | Descriptive tags (attributes, clothing, composition, etc.) |
| `rating` | Top rating tag (safe / sensitive / questionable / explicit) |
| `character_tags` | Character, copyright, and artist tags |

## Models

| Model | Parameters | Tags | Input Size | License | Gated |
|---|---|---|---|---|---|
| WD Series (eva02, vit, swinv2, etc.) | varies | varies | 448² | MIT | No |
| Pixai Tagger v0.9 | — | 13,461 | 448² | Apache-2.0 | No |
| Camie Tagger v2 | — | 70,527 | 512² | ? | No |
| CL Tagger v1 (1.00 / 1.01 / 1.02) | — | 42,163 | 448² | Apache-2.0 | No |
| CL Tagger v2 (2.00 / 2.01a) | — | 106,536 / 108,036 | 384² | Custom | **Yes** |
| AnimeTimm swinv2_base | 99.7M | 12,476 | 256² | GPL-3.0 | **Yes** |
| AnimeTimm caformer_b36 | 134.0M | 12,476 | 384² | GPL-3.0 | **Yes** |
| AnimeTimm eva02_large | 316.8M | 12,476 | 448² | GPL-3.0 | **Yes** |
| AnimeTimm ConvNeXtV2 Huge (community ONNX) | 692.6M | 12,476 | 512² | GPL-3.0 | **Yes** |

> ConvNeXtV2 Huge uses the community ONNX conversion from [itterative](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx), with official AnimeTimm metadata and preprocessing.

> **Gated models require a HuggingFace token.** Accept the license on the model page, then either run `huggingface-cli login` or set the `HF_TOKEN` environment variable before first download.

Credits:
- [pythongosssss/ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)
- [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags)
- [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)

Models created by:
- WD Taggers: [SmilingWolf](https://huggingface.co/SmilingWolf)
- Pixai Tagger: [pixai-labs](https://huggingface.co/pixai-labs)
- Camie Tagger: [Camais03](https://huggingface.co/Camais03)
- CL Tagger v1 / v2: [cella110n](https://huggingface.co/cella110n)
- AnimeTimm: [DeepGHS](https://huggingface.co/deepghs) / [narugo1992](https://huggingface.co/narugo1992)

## Installation
1. Clone this repo into the `custom_nodes` folder.
2. Install dependency (`onnxruntime` or `onnxruntime-gpu` for CUDA acceleration).
3. For gated models (CL Tagger v2, AnimeTimm): run `huggingface-cli login` once, or set the `HF_TOKEN` environment variable with your HuggingFace token.

## Configuration

Edit `models.json` to customize defaults:

- **`settings.ortProviders`** — ONNX Runtime execution providers. GPU is auto-detected.
- **`threshold` / `character_threshold`** — Per-model default thresholds.
- **`HF_ENDPOINT`** — Mirror/proxy URL for HuggingFace downloads.
