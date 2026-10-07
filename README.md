# ComfyUI Booru Tagger

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension that looks at an image and writes booru-style tags for it (`1girl, long hair, smile, ...`), ready to use in a prompt. It bundles 25 open tagger models from five families — [WD](https://huggingface.co/SmilingWolf), [Pixai](https://huggingface.co/pixai-labs/pixai-tagger-v1.0), [Camie](https://huggingface.co/Camais03/camie-tagger-v2), [CL Tagger](https://huggingface.co/cella110n) and [AnimeTimm](https://huggingface.co/animetimm) — and downloads them for you on first use.

- **Many models, one node.** Switch taggers from a dropdown; each model's own preprocessing and recommended thresholds are applied for you.
- **Batches.** Tag a whole image batch in one run and get one tag string per image.
- **Separate tag groups.** General, character (with copyright and artist) and rating come out as separate outputs.
- **Fast on GPU.** On CUDA/ROCm, images are prepared on the GPU and passed straight to the model.

## Installation

1. Install **Booru Tagger** from ComfyUI-Manager or the [Comfy Registry](https://registry.comfy.org/nodes/booru-tagger) (`comfy node install booru-tagger`). Dependencies are installed automatically.

   Or clone this repo into `ComfyUI/custom_nodes` and run `pip install -r requirements.txt` in ComfyUI's Python environment.
2. Restart ComfyUI.
3. Optional: to use the gated models (CL Tagger v2 and AnimeTimm), see [Gated models](#gated-models).

## Quick start

1. Add **Load Booru Tagger** and pick a model.
2. Add **Booru Tagger** and connect `tagger_model`, `tagger_info`, `threshold` and `character_threshold` from the loader, plus your `IMAGE`.
3. Send `tags` to a text/preview node or straight into a prompt.

The first run downloads the model into `ComfyUI/models/booru_tagger/<model name>/`, which can take a while for the large ones. Later runs reuse the files.

## Nodes

### Load Booru Tagger

Loads a model and its tag list, downloading them if missing.

| Output | Description |
|---|---|
| `tagger_model`, `tagger_info` | Connect both to Booru Tagger. |
| `threshold`, `character_threshold` | The model's recommended thresholds. Connect them to Booru Tagger to use them. |

### Booru Tagger

| Input | Default | Description |
|---|---|---|
| `image` | | Image or image batch to tag. |
| `threshold` | 0.35 | Minimum confidence for general tags. Lower gives more tags, higher gives fewer but surer ones. |
| `character_threshold` | 0.85 | Minimum confidence for character, copyright and artist tags. |
| `use_best_threshold` | on | Also apply the per-tag thresholds shipped with AnimeTimm and Pixai v1.0. Turn off to rely only on the two thresholds above. Other models ignore it. |
| `trailing_comma` | off | End every tag with a comma (`tag1, tag2, `), useful when concatenating strings. |
| `sort_tags` | off | Order tags by confidence instead of the model's tag order. |
| `exclude_tags` | | Comma-separated tags to drop, e.g. `simple background, white background`. Case, underscores and spaces don't matter. |
| `chunk_size` | 0 | How many images go through the model at once. 0 = whole batch. Lower it if you run out of VRAM. |
| `replace_underscore` | on | Write `long hair` instead of `long_hair`. |

If `threshold` and `character_threshold` are left unconnected at their defaults, the model's recommended values are used instead, with a one-time note in the console.

| Output | Description |
|---|---|
| `tags` | Character tags followed by general tags. Usually what you want for a prompt. |
| `general_tags` | Descriptive tags: appearance, clothing, pose, background, composition. |
| `character_tags` | Character, copyright (series) and artist tags. |
| `rating` | The single most likely rating. The wording depends on the model, e.g. `general` / `sensitive` / `questionable` / `explicit` for WD. |

Each output is a list with one string per input image. Parentheses are escaped (`star \(symbol\)`) so tags can be pasted into a ComfyUI prompt without being read as weights.

### Unique Tags

Removes duplicate tags from a comma-separated string, keeping the first occurrence. Handy after joining the output of several taggers.

## Choosing a model

Any model works for general use. Some pointers:

- **Fast and simple:** WD v3 models (`wd-eva02-large-tagger-v3` is the largest, `wd-vit-tagger-v3` and `wd-convnext-tagger-v3` are lighter). No login needed.
- **Newest and largest vocabulary:** Pixai Tagger v1.0 (30,877 tags) and CL Tagger. Pixai v1.0 runs at 1008×1008, so it is heavy: use the FP16 variant on a GPU, FP32 on CPU, and set `chunk_size` if VRAM is tight.
- **Calibrated per-tag thresholds:** AnimeTimm models, with `use_best_threshold` on.

| Model | Tags | Input size | License | Login needed |
|---|---|---|---|---|
| WD v3: `wd-eva02-large-tagger-v3`, `wd-vit-large-tagger-v3`, `wd-vit-tagger-v3`, `wd-swinv2-tagger-v3`, `wd-convnext-tagger-v3` | 10,861 | 448² | Apache-2.0 | No |
| WD v1.4 v2: `moat`, `convnextv2`, `convnext`, `vit`, `swinv2` | 9,083 | 448² | Apache-2.0 | No |
| WD v1.4 v1: `convnext`, `vit` | 6,549 | 448² | see model page | No |
| Pixai Tagger v0.9 | 13,461 | 448² | Apache-2.0 | No |
| Pixai Tagger v1.0 (FP16 / FP32) | 30,877 | 1008² | Apache-2.0 | No |
| Camie Tagger v2 | 70,527 | 512² | GPL-3.0 | No |
| CL Tagger v1 (1.00 / 1.01 / 1.02) | 42,163 | 448² | Apache-2.0 | No |
| CL Tagger v2 (2.00 / 2.01a) | 106,536 / 108,036 | 384² | Custom | **Yes** |
| AnimeTimm swinv2_base | 12,476 | 256² | GPL-3.0 | **Yes** |
| AnimeTimm caformer_b36 | 12,476 | 384² | GPL-3.0 | **Yes** |
| AnimeTimm eva02_large | 12,476 | 448² | GPL-3.0 | **Yes** |
| AnimeTimm ConvNeXtV2 Huge | 12,476 | 512² | GPL-3.0 | **Yes** |

Notes:

- Pixai Tagger v1.0 uses the community ONNX conversion by [Mexes](https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8). Its artist (called "style" on the model card) and copyright tags go to `character_tags`. With `use_best_threshold` on, each category keeps the model card's threshold: general 0.17, artist 0.15, copyright 0.24, character 0.27.
- AnimeTimm ConvNeXtV2 Huge uses the community ONNX conversion by [itterative](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx) with the official AnimeTimm tag list and preprocessing.
- Recommended AnimeTimm thresholds (general / character): eva02 0.39 / 0.61, caformer 0.39 / 0.47, swinv2 0.41 / 0.59, ConvNeXtV2 Huge 0.38 / 0.51.

## Gated models

CL Tagger v2 and the AnimeTimm models require accepting their license on Hugging Face before download:

1. Open the model page (linked above), sign in and accept the terms.
2. Run `hf auth login` once in ComfyUI's Python environment, or set the `HF_TOKEN` environment variable to your token.
3. Restart ComfyUI and run the loader again.

## Configuration

Defaults live in `models.json` in the extension folder. Updating the extension may overwrite your edits, so keep a copy.

| Setting | Description |
|---|---|
| `settings.model` | Model selected by default in the loader. |
| `settings.threshold`, `settings.character_threshold` | Global default thresholds. |
| `settings.exclude_tags` | Default for `exclude_tags`. |
| `settings.ortProviders` | ONNX Runtime providers in order of preference. Providers your onnxruntime build lacks are skipped with a warning. |
| `settings.preprocess` | `"tensor"` (default) prepares images on the GPU; `"pil"` uses the CPU/PIL path. Both give the same tags. |
| `settings.ortIoBinding` | Pass GPU-prepared images to the model without copying them back to the CPU (CUDA/ROCm only). Falls back automatically if unsupported. |
| `settings.ortIntraOpThreads` | CPU threads for ONNX Runtime. 0 = automatic. |
| `settings.HF_ENDPOINT` | Hugging Face mirror for downloads. The `HF_ENDPOINT` environment variable takes priority. |
| `threshold`, `character_threshold` | Per-model recommended thresholds, returned by the loader. |
| `logging` | `true` prints extra debug messages. |

## Troubleshooting

- **401 / authentication error while downloading:** the model is gated. See [Gated models](#gated-models).
- **Downloads fail or are slow:** set `HF_ENDPOINT` to a Hugging Face mirror, or download the files manually into `ComfyUI/models/booru_tagger/<model name>/` using the file names in `models.json` (`model_path`, `metadata_path`). Your Hugging Face token is only sent to huggingface.co, so gated models still need a direct connection.
- **Out of memory:** lower `chunk_size`, or choose a smaller model (or Pixai v1.0 FP16 instead of FP32).
- **Runs on CPU although you have an NVIDIA GPU:** check the `Using ORT providers` line in the console. If `CUDAExecutionProvider` is missing, the installed `onnxruntime-gpu` does not match your CUDA setup.
- **Too many / too few tags:** adjust `threshold` and `character_threshold`, or toggle `use_best_threshold`.
- **Upgrading from ComfyUI-WD14-Tagger:** model files found in `ComfyUI/models/wd14_tagger` are moved to the new location automatically.

## Credits

Based on [pythongosssss/ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger), with ideas from [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) and [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger).

Models by [SmilingWolf](https://huggingface.co/SmilingWolf) (WD), [pixai-labs](https://huggingface.co/pixai-labs) (Pixai), [Camais03](https://huggingface.co/Camais03) (Camie), [cella110n](https://huggingface.co/cella110n) (CL Tagger) and [DeepGHS](https://huggingface.co/deepghs) / [narugo1992](https://huggingface.co/narugo1992) (AnimeTimm).
