from comfy_api.latest import ComfyExtension, io
import numpy as np
import asyncio
import os
import aiohttp
import folder_paths
import sys
import onnxruntime
# from server import PromptServer
from aiohttp import web
from PIL import Image
from .utils import get_ext_dir, download_to_file, get_extension_config, log
from onnxruntime import InferenceSession
from typing_extensions import override
from comfy import utils
import pandas as pd
import json
import torchvision.transforms as transforms
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))

config = get_extension_config()

defaults = {
    "model": "wd-eva02-large-tagger-v3",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": True,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co"
}
defaults.update(config.get("settings", {}))

# Filter ORT providers: try GPU first, fall back to CPU. Excludes AzureExecutionProvider (deprecated shim)
# and TensorrtExecutionProvider (crashes without full NVIDIA TensorRT SDK installed).
_ORT_BLOCKLIST = {"AzureExecutionProvider", "TensorrtExecutionProvider"}
available_providers = set(onnxruntime.get_available_providers())
_ORT_PRIORITY = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]
defaults["ortProviders"] = [p for p in _ORT_PRIORITY if p in available_providers and p not in _ORT_BLOCKLIST]
if not defaults["ortProviders"]:
    defaults["ortProviders"] = ["CPUExecutionProvider"]

folder_name = "booru_tagger"
target_folder_path = os.path.join(folder_paths.models_dir, folder_name)
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path, exist_ok=True)
folder_paths.add_model_folder_path(folder_name, target_folder_path)
models_dir = folder_paths.get_folder_paths(folder_name)[0]

# Directories where legacy flat files may exist (v1.x used "wd14_tagger" or extension-local "models")
_LEGACY_MODEL_DIRS = [
    os.path.join(folder_paths.models_dir, "wd14_tagger"),
    get_ext_dir("models"),
]
known_models = list(config["model_url"].keys())

log("Available ORT providers: " +
    ", ".join(onnxruntime.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " +
    ", ".join(defaults["ortProviders"]), "DEBUG", True)

def _migrate_legacy_model(model_name, dest_model, dest_meta):
    """Move legacy flat files into the nested directory structure (v1.x → v2.x).

    Before v2.x, files were stored flat in one of:
      - ComfyUI/models/wd14_tagger/<model>.onnx
      - <extension>/models/<model>.onnx

    After v2.x, files are stored nested:
      - ComfyUI/models/booru_tagger/<model>/model.onnx
    """
    if os.path.exists(dest_model) and os.path.exists(dest_meta):
        return  # Already migrated or downloaded fresh

    # Scan all possible legacy locations
    for legacy_dir in _LEGACY_MODEL_DIRS:
        if not os.path.isdir(legacy_dir):
            continue
        legacy_model = os.path.join(legacy_dir, model_name + ".onnx")
        if not os.path.exists(legacy_model):
            continue

        legacy_csv = os.path.join(legacy_dir, model_name + ".csv")
        legacy_json = os.path.join(legacy_dir, model_name + ".json")

        os.makedirs(os.path.dirname(dest_model), exist_ok=True)
        if not os.path.exists(dest_model):
            os.rename(legacy_model, dest_model)

        for legacy_meta in (legacy_csv, legacy_json):
            if os.path.exists(legacy_meta):
                os.makedirs(os.path.dirname(dest_meta), exist_ok=True)
                if not os.path.exists(dest_meta):
                    os.rename(legacy_meta, dest_meta)
                break

        log(f"Migrated legacy files for {model_name} from {legacy_dir} to nested layout", "INFO", True)
        return


def wd_tag(wd_model: InferenceSession, img: Image.Image):
    img_input = wd_model.get_inputs()[0]
    (batch_size, height, width, channel) = img_input.shape

    # Reduce to max size and pad with white
    ratio = float(height)/max(img.size)
    new_size = tuple([int(x*ratio) for x in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    pad_color = (255, 255, 255)
    new_img = Image.new("RGB", (height, height), pad_color)
    paste_x = (height-new_size[0]) // 2
    paste_y = (height-new_size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))

    img_numpy = np.array(new_img, dtype=np.float32)
    img_numpy = img_numpy[:, :, ::-1]  # RGB -> BGR
    img_numpy = np.expand_dims(img_numpy, 0)  # Batch dim

    label_name = wd_model.get_outputs()[0].name
    probs = wd_model.run([label_name], {img_input.name: img_numpy})[0]
    result = probs[0]

    return result


def pixai_tag(pixai_model: InferenceSession, img):
    img_input = pixai_model.get_inputs()[0]
    (batch_size, channel, height, width) = img_input.shape

    # Reduce to max size and pad with white
    ratio = float(height)/max(img.size)
    new_size = tuple([int(x*ratio) for x in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    pad_color = (128, 128, 128)
    new_img = Image.new("RGB", (height, height), pad_color)
    paste_x = (height-new_size[0]) // 2
    paste_y = (height-new_size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    img_tensor = transform(new_img)
    img_numpy = torch.unsqueeze(img_tensor, 0).numpy()

    pred_name = pixai_model.get_outputs()[2].name
    prediction = pixai_model.run([pred_name], {img_input.name: img_numpy})[0]
    result = prediction[0]
    return result


def camie_tag(camie_model: InferenceSession, img):
    img_input = camie_model.get_inputs()[0]
    (batch_size, channel, height, width) = img_input.shape

    # Reduce to max size and pad with white
    ratio = float(height)/max(img.size)
    new_size = tuple([int(x*ratio) for x in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    pad_color = (124, 116, 104)
    new_img = Image.new("RGB", (height, height), pad_color)
    paste_x = (height-new_size[0]) // 2
    paste_y = (height-new_size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(new_img)
    img_numpy = torch.unsqueeze(img_tensor, 0).numpy()

    init_pred_name = camie_model.get_outputs()[0].name
    refine_pred_name = camie_model.get_outputs()[1].name
    select_cand_name = camie_model.get_outputs()[2].name
    (init_logits, ref_logits, select_cands) = camie_model.run(
        [init_pred_name, refine_pred_name, select_cand_name], {img_input.name: img_numpy})

    probs = 1.0 / (1.0 + np.exp(-ref_logits))
    result = probs[0]
    return result


def animetimm_tag(animetimm_model: InferenceSession, img: Image.Image, preprocess: dict):
    """Inference for animetimm timm-based taggers (dbv4-full family).

    Preprocessing pipeline matches the model's preprocess.json:
      PadToSize → Resize → CenterCrop → Normalize
    Mean/std and sizes are per-backbone, read from the downloaded preprocess.json.
    """
    # Extract pipeline params from preprocess.json (test split)
    steps = preprocess["test"]
    pad_step = next(s for s in steps if s["type"] == "pad_to_size")
    resize_step = next(s for s in steps if s["type"] == "resize")
    crop_step = next(s for s in steps if s["type"] == "center_crop")
    norm_step = next(s for s in steps if s["type"] == "normalize")

    pad_size = tuple(pad_step["size"])          # e.g. (512, 512)
    resize_size = resize_step["size"]            # int (shorter side) or [int, int] (exact)
    crop_size = crop_step["size"]                # [int, int]
    mean = norm_step["mean"]
    std = norm_step["std"]

    # PadToSize: pad shorter side with white so both dims ≥ pad_size
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]
    if h < pad_size[0] or w < pad_size[1]:
        new_h = max(h, pad_size[0])
        new_w = max(w, pad_size[1])
        padded = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
        off_h = (new_h - h) // 2
        off_w = (new_w - w) // 2
        padded[off_h:off_h+h, off_w:off_w+w] = img_np
        img = Image.fromarray(padded)

    # PadToSize → Resize → CenterCrop → Normalize  (exactly as preprocess.json)
    transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img_tensor = transform(img)
    img_numpy = torch.unsqueeze(img_tensor, 0).numpy()

    # Run all outputs — timm ONNX exports may have multiple heads (embedding + logits).
    # Pick the output with the largest last dimension (the tag logits).
    img_input = animetimm_model.get_inputs()[0]
    output_names = [o.name for o in animetimm_model.get_outputs()]
    outputs = animetimm_model.run(output_names, {img_input.name: img_numpy})
    best_idx = max(range(len(outputs)), key=lambda i: outputs[i].shape[-1])
    logits = outputs[best_idx]
    probs = 1.0 / (1.0 + np.exp(-logits))
    result = probs[0]
    return result


def _load_animetimm_preprocess(model_name: str) -> dict:
    """Load the per-model preprocess.json shipped alongside the ONNX file."""
    path = os.path.join(models_dir, f"{model_name}.preprocess.json")
    if not os.path.exists(path):
        log(f"No preprocess.json found for {model_name}, using fallback", "WARN", True)
        return {
            "test": [
                {"type": "pad_to_size", "size": [512, 512]},
                {"type": "resize", "size": [448, 448]},
                {"type": "center_crop", "size": [448, 448]},
                {"type": "normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            ]
        }
    with open(path) as f:
        return json.load(f)


def cl_tagger_v2_tag(cl_model: InferenceSession, img: Image.Image):
    img_input = cl_model.get_inputs()[0]
    (batch_size, channel, height, width) = img_input.shape

    # SigLIP2 preprocessing: direct resize to 384x384, normalize mean=std=0.5
    img = img.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)

    img_numpy = np.asarray(img, dtype=np.float32) / 255.0
    img_numpy = (img_numpy - 0.5) / 0.5
    img_numpy = img_numpy.transpose(2, 0, 1)[None]  # [1, 3, H, W]

    logits_name = cl_model.get_outputs()[0].name
    logits = cl_model.run([logits_name], {img_input.name: img_numpy})[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    result = probs[0]

    return result


def cl_tagger_v1_tag(cl_model: InferenceSession, img: Image.Image):
    img_input = cl_model.get_inputs()[0]
    input_shape = img_input.shape

    # Detect layout from raw shape (may contain strings for dynamic dims)
    # NCHW: [batch, 3, H, W]   NHWC: [batch, H, W, 3]
    is_nchw = len(input_shape) == 4 and input_shape[1] == 3
    is_nhwc = len(input_shape) == 4 and input_shape[3] == 3

    # Determine target size, fall back to 448
    target_size = 448
    if is_nchw:
        for idx in (3, 2):  # prefer W, then H
            if isinstance(input_shape[idx], int):
                target_size = input_shape[idx]
                break
    elif is_nhwc:
        for idx in (2, 1):  # prefer W, then H
            if isinstance(input_shape[idx], int):
                target_size = input_shape[idx]
                break

    # Pad to square with white (same as wd-eva02-large-tagger-v3 preprocessing)
    width, height = img.size
    if width != height:
        new_size = max(width, height)
        new_img = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        paste_x = (new_size - width) // 2
        paste_y = (new_size - height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img

    # Resize to target size
    img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)

    img_numpy = np.asarray(img, dtype=np.float32) / 255.0
    img_numpy = img_numpy[:, :, ::-1]  # RGB -> BGR

    if is_nchw:
        img_numpy = img_numpy.transpose(2, 0, 1)  # HWC -> CHW
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    else:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    img_numpy = (img_numpy - mean) / std
    img_numpy = np.expand_dims(img_numpy, 0)  # [1, C, H, W] or [1, H, W, C]

    logits_name = cl_model.get_outputs()[0].name
    logits = cl_model.run([logits_name], {img_input.name: img_numpy})[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    result = probs[0]

    return result


def _format_tags(tag_list, trailing_comma=False):
    if not tag_list:
        return ""
    res = ("" if trailing_comma else ", ").join((tag.replace(
        "(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for tag in tag_list))
    return res


def _pick_top_rating(df):
    """Pick the single highest-probability tag from rating category (1)."""
    rating_rows = df[(df['category'] == 1) & (df['probs'] > 0)]
    if rating_rows.empty:
        return ""
    top = rating_rows.loc[rating_rows['probs'].idxmax()]
    return top['name']


# Category convention for model loaders:
#   0 = general (descriptive tags, shown in general_tags)
#   1 = rating  (4 tags: safe/sensitive/questionable/explicit, shown in rating output)
#   2 = quality (4 tags: best/normal/bad/worst, excluded from general/character)
#   3 = meta    (metadata / model info, excluded from general/character)
#   4 = character (character/copyright/artist, shown in character_tags)


def get_tag(probs, tags_df: pd.DataFrame, threshold=0.35, character_threshold=0.85, trailing_comma=False, sort_tags=False, exclude_tags=""):
    df = tags_df.assign(probs=probs)
    if sort_tags:
        df = df.sort_values(by='probs', ascending=False)

    # Use per-tag best_threshold when available (animetimm models), else global threshold.
    # Per-tag thresholds act as a floor; the user's slider can raise the bar further.
    if 'best_threshold' in df.columns:
        best = df['best_threshold'].fillna(1.0)
        general = df[(df['category'] == 0) & (df['probs'] >= np.maximum(best, threshold))]['name'].to_list()
        character = df[(df['category'] == 4) & (df['probs'] >= np.maximum(best, character_threshold))]['name'].to_list()
    else:
        general = df[(df['category'] == 0) & (df['probs'] > threshold)]['name'].to_list()
        character = df[(df['category'] == 4) & (df['probs'] > character_threshold)]['name'].to_list()
    rating = _pick_top_rating(df)

    remove = [s.strip() for s in exclude_tags.lower().split(",")] if exclude_tags else []

    def _apply_exclude(tag_list):
        if not remove:
            return tag_list
        return [tag for tag in tag_list if tag.lower() not in remove]

    character = _apply_exclude(character)
    general = _apply_exclude(general)
    tags = character + general

    return {
        "combined": _format_tags(tags, trailing_comma),
        "rating": rating,
        "character": _format_tags(character, trailing_comma),
        "general": _format_tags(general, trailing_comma),
    }


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["model_url"][model]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main"

    model_path = config["model_path"].get(model, model + ".onnx")
    metadata_path = config["metadata_path"].get(model, "selected_tags.csv")
    external_data_path = config.get("external_data_path", {}).get(model, None)
    dest_model_path = os.path.join(models_dir, model_path)
    dest_metadata_path = os.path.join(models_dir, metadata_path)

    # 优先读取 models.json 中的 remote 映射关系，缺省则取本地路径的 basename
    remote_model_path = config.get("remote_model_path", {}).get(model, os.path.basename(model_path))
    remote_metadata_path = config.get("remote_metadata_path", {}).get(model, os.path.basename(metadata_path))

    # Support HF token for gated models.
    # Priority: HF_TOKEN env var → HUGGINGFACE_TOKEN env var → huggingface_hub cache (hf auth login)
    hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with aiohttp.ClientSession(loop=asyncio.get_event_loop(), headers=headers) as session:
        async def update_callback(perc):
            nonlocal client_id

        try:
            # Only download the ONNX model file if it does not exist locally
            if not os.path.exists(dest_model_path):
                os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)
                log(f"Downloading model {model} to {dest_model_path}...", "INFO", True)
                await download_to_file(
                    f"{url}/{remote_model_path}", dest_model_path, update_callback, session=session)

            # Only download external data file if required and missing
            if external_data_path:
                dest_ext_path = os.path.join(models_dir, external_data_path)
                if not os.path.exists(dest_ext_path):
                    os.makedirs(os.path.dirname(dest_ext_path), exist_ok=True)
                    log(f"Downloading external data file for {model} to {dest_ext_path}...", "INFO", True)
                    await download_to_file(
                        f"{url}/{external_data_path}", dest_ext_path, update_callback, session=session)

            # Only download the metadata file if it does not exist locally
            if not os.path.exists(dest_metadata_path):
                os.makedirs(os.path.dirname(dest_metadata_path), exist_ok=True)
                log(f"Downloading metadata to {dest_metadata_path}...", "INFO", True)
                await download_to_file(
                    f"{url}/{remote_metadata_path}", dest_metadata_path, update_callback, session=session)

            # Only download preprocess.json if required (animetimm models)
            preprocess_path = config.get("preprocess_path", {}).get(model, None)
            if preprocess_path:
                dest_preprocess_path = os.path.join(models_dir, preprocess_path)
                if not os.path.exists(dest_preprocess_path):
                    os.makedirs(os.path.dirname(dest_preprocess_path), exist_ok=True)
                    log(f"Downloading preprocess for {model} to {dest_preprocess_path}...", "INFO", True)
                    await download_to_file(
                        f"{url}/preprocess.json", dest_preprocess_path, update_callback, session=session)

            # Download preprocess.json for animetimm models (per-backbone normalization)
            if model.startswith("animetimm"):
                await download_to_file(
                    f"{url}/preprocess.json", os.path.join(models_dir, f"{model}.preprocess.json"), update_callback, session=session)

        except aiohttp.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise
        except aiohttp.ClientError as err:
            status = getattr(err, 'status', None)
            if status == 401:
                log(f"Authentication required for {model}. Run `hf auth login` or set the HF_TOKEN environment variable with your HuggingFace token.", "ERROR", True)
            else:
                log(f"Download failed: {err}", "ERROR", True)
            raise
        except Exception as err:
            log(f"Download failed: {err}", "ERROR", True)
            raise

    return web.Response(status=200)


class BooruTagger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Booru Tagger",
            category="BooruTagger",
            inputs=[
                io.Custom("TAGGER_MODEL").Input("tagger_model"),
                io.Custom("TAGGER_INFO").Input("tagger_info"),
                io.Image.Input("image"),
                io.Float.Input("threshold", min=0.0, max=1.0,
                               step=0.05, default=defaults["threshold"]),
                io.Float.Input("character_threshold",
                               min=0.0, max=1.0, step=0.05, default=defaults["character_threshold"]),
                io.Boolean.Input("trailing_comma",
                                 default=defaults["trailing_comma"]),
                io.Boolean.Input("sort_tags", default=False),
                io.String.Input(
                    "exclude_tags", default=defaults["exclude_tags"], multiline=True),
            ],
            outputs=[
                io.String.Output("tags", is_output_list=True),
                io.String.Output("general_tags", is_output_list=True),
                io.String.Output("rating", is_output_list=True),
                io.String.Output("character_tags", is_output_list=True),
            ]
        )

    @classmethod
    def execute(cls, tagger_model, tagger_info, image, threshold, character_threshold, trailing_comma=False, sort_tags=False, exclude_tags="") -> io.NodeOutput:
        pbar = utils.ProgressBar(image.shape[0])
        tags, ratings, chara_tags, general_tags = [], [], [], []
        for i in range(image.shape[0]):
            img = Image.fromarray(np.array(image[i] * 255, dtype=np.uint8))

            tags_df = tagger_info[0]
            model_name = tagger_info[1]

            if model_name.startswith("pixai-tagger"):
                probs = pixai_tag(tagger_model, img)
            elif model_name.startswith("camie-tagger-v2"):
                probs = camie_tag(tagger_model, img)
            elif model_name.startswith("animetimm"):
                preprocess = _load_animetimm_preprocess(model_name)
                probs = animetimm_tag(tagger_model, img, preprocess)
            elif model_name.startswith("cl-tagger-v1"):
                probs = cl_tagger_v1_tag(tagger_model, img)
            elif model_name.startswith("cl-tagger-v2"):
                probs = cl_tagger_v2_tag(tagger_model, img)
            else:  # WD tagger
                probs = wd_tag(tagger_model, img)

            result = get_tag(probs, tags_df, threshold,
                             character_threshold, trailing_comma, sort_tags, exclude_tags)
            tags.append(result["combined"])
            ratings.append(result["rating"])
            chara_tags.append(result["character"])
            general_tags.append(result["general"])
            pbar.update(1)
        return io.NodeOutput(tags, general_tags, ratings, chara_tags)


class LoadBooruTaggerModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        models = known_models
        return io.Schema(
            node_id="Load Booru Tagger",
            category="BooruTagger",
            inputs=[
                io.Combo.Input("model_name", options=models,
                               default=defaults["model"]),
                io.Boolean.Input("replace_underscore",
                                 default=defaults["replace_underscore"]),
            ],
            outputs=[
                io.Custom("TAGGER_MODEL").Output("tagger_model"),
                io.Custom("TAGGER_INFO").Output("tagger_info"),
                io.Float.Output("threshold"),
                io.Float.Output("character_threshold")
            ]
        )

    @classmethod
    async def execute(cls, model_name, replace_underscore, client_id=None, node=None) -> io.NodeOutput:
        # Get paths directly from models.json config to avoid scanning guessworks
        rel_model_path = config["model_path"].get(model_name)
        if not rel_model_path:
            raise ValueError(f"Model path for {model_name} is not defined in models.json")
        name = os.path.join(models_dir, rel_model_path)

        rel_metadata_path = config["metadata_path"].get(model_name)
        if not rel_metadata_path:
            raise ValueError(f"Metadata path for {model_name} is not defined in models.json")
        meta_path = os.path.join(models_dir, rel_metadata_path)

        # Migrate legacy flat files to nested structure (from versions < 2.x)
        _migrate_legacy_model(model_name, name, meta_path)

        # Download if any required file is missing
        needs_download = not os.path.exists(name) or not os.path.exists(meta_path)
        preprocess_path = config.get("preprocess_path", {}).get(model_name)
        if preprocess_path and not os.path.exists(os.path.join(models_dir, preprocess_path)):
            needs_download = True
        if needs_download:
            await download_model(model_name, client_id, node)

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3  # Suppress provider init warnings
        model = InferenceSession(name, sess_options=sess_options, providers=defaults["ortProviders"])

        threshold = config["threshold"].get(model_name, defaults["threshold"])
        character_threshold = config["character_threshold"].get(model_name, defaults["character_threshold"])

        # Validate that metadata actually exists after the download step
        if not os.path.exists(meta_path):
            log(f"No tag data is found for {model_name} at: {meta_path}")
            raise FileNotFoundError(f"Required tag metadata file is missing: {meta_path}")

        if (model_name.startswith("wd") or model_name.startswith("pixai") or model_name.startswith("animetimm")) and meta_path.endswith(".csv"):
            df = pd.read_csv(meta_path)
            # Remap WD rating tags from category 9 → 1 (rating)
            df.loc[df['category'] == 9, 'category'] = 1
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            preprocess = _load_animetimm_preprocess(model_name) if model_name.startswith("animetimm") else None
            return io.NodeOutput(model, (df, model_name, preprocess), threshold, character_threshold)
            
        elif model_name.startswith("camie") and meta_path.endswith(".json"):
            df = pd.DataFrame()
            with open(meta_path, encoding="utf-8") as f:
                js = json.load(f)
                tag_mapping = js["dataset_info"]["tag_mapping"]
                df["name"] = list(tag_mapping["idx_to_tag"].values())
                df["category_name"] = list(
                    tag_mapping["tag_to_category"].values())
                _cat_map_camie = {
                    "general": 0, "rating": 1,
                    "meta": 3, "year": 3,
                    "character": 4, "artist": 4, "copyright": 4,
                }
                df["category"] = df["category_name"].map(
                    lambda c: _cat_map_camie.get(c, 0))
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            return io.NodeOutput(model, (df, model_name), threshold, character_threshold)
            
        elif model_name.startswith("cl-tagger-v1") and meta_path.endswith(".json"):
            df = pd.DataFrame()
            with open(meta_path, encoding="utf-8") as f:
                js = json.load(f)

                if "idx_to_tag" in js:
                    idx_to_tag = js["idx_to_tag"]
                    tag_to_category = js["tag_to_category"]
                else:
                    idx_to_tag = {}
                    tag_to_category = {}
                    for k, v in js.items():
                        idx = int(k)
                        tag_name = v["tag"]
                        idx_to_tag[str(idx)] = tag_name
                        tag_to_category[tag_name] = v["category"]

                _cat_map_v1 = {
                    "general": 0, "rating": 1, "quality": 2,
                    "meta": 3, "model": 3,
                    "character": 4, "copyright": 4, "artist": 4,
                }

                tag_names = []
                tag_cats = []
                for idx_str in sorted(idx_to_tag.keys(), key=int):
                    tag_name = idx_to_tag[idx_str]
                    category = tag_to_category.get(tag_name, "").lower()
                    tag_names.append(tag_name)
                    tag_cats.append(_cat_map_v1.get(category, 0))

                df["name"] = tag_names
                df["category"] = tag_cats
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            return io.NodeOutput(model, (df, model_name), threshold, character_threshold)
            
        elif model_name.startswith("cl-tagger-v2") and meta_path.endswith(".json"):
            df = pd.DataFrame()
            with open(meta_path, encoding="utf-8") as f:
                js = json.load(f)
                idx_to_tag = js["idx_to_tag"]
                tag_to_category = js["tag_to_category"]
                categories = js["categories"]

                _cat_map_v2 = {
                    "general": 0, "rating": 1, "quality": 2,
                    "meta": 3,
                    "character": 4, "copyright": 4,
                }

                tag_names = []
                tag_cats = []
                for idx_str in sorted(idx_to_tag.keys(), key=int):
                    tag_name = idx_to_tag[idx_str]
                    category = tag_to_category.get(tag_name, "").lower()
                    tag_names.append(tag_name)
                    tag_cats.append(_cat_map_v2.get(category, 0))

                df["name"] = tag_names
                df["category"] = tag_cats
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            return io.NodeOutput(model, (df, model_name), threshold, character_threshold)
        else:
            log("No compatible tag data parser found for this model.")
            # Raise an exception instead of calling exit(1) to prevent ComfyUI from freezing
            raise ValueError(f"No compatible tag data parser found for model: {model_name}")

class UniqueTags(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Unique Tags",
            category="BooruTagger",
            inputs=[
                io.String.Input("input_tags")
            ],
            outputs=[
                io.String.Output("tags"),
            ]
        )

    @classmethod
    def execute(cls, input_tags) -> io.NodeOutput:
        unique_tags = []
        for tag in input_tags.split(','):
            tag = tag.strip()
            if len(tag) > 0 and tag not in unique_tags:
                unique_tags.append(tag)

        unique_tags = ', '.join(unique_tags)
        return io.NodeOutput(unique_tags)


class BooruTaggerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadBooruTaggerModel,
            BooruTagger,
            UniqueTags
        ]
