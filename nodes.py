from comfy_api.latest import ComfyExtension, io
import numpy as np
import os
import aiohttp
import folder_paths
import onnxruntime
from PIL import Image
from .utils import get_ext_dir, download_to_file, get_extension_config, log
from onnxruntime import InferenceSession
from typing_extensions import override
from comfy import utils
import pandas as pd
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

config = get_extension_config()
known_models = list(config.get("model_url", {}).keys())

# models.json is the single source of truth for defaults (see its "settings"
# section). The values below are only fallbacks for keys a config file does not
# provide; the default model is derived from model_url so the two can never
# drift apart again.
defaults = {
    "model": known_models[0] if known_models else "wd-eva02-large-tagger-v3",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": True,
    "trailing_comma": False,
    "exclude_tags": "",
    "HF_ENDPOINT": "https://huggingface.co",
    # "tensor" resizes/pads/normalises on the GPU straight from ComfyUI's IMAGE
    # tensor; "pil" is the original PIL path over PIL-resized images.
    "preprocess": "tensor",
}
defaults.update(config.get("settings", {}))

_PREPROCESS = defaults["preprocess"]
if _PREPROCESS not in ("tensor", "pil"):
    log(f"Unknown settings.preprocess {_PREPROCESS!r}, using 'tensor'", "WARN", True)
    _PREPROCESS = "tensor"

# Guard against a stale default model (e.g. settings.model was removed from
# model_url during a config edit): fall back to the first configured model.
if defaults["model"] not in known_models:
    if known_models:
        log(f"Default model {defaults['model']!r} is not listed in models.json model_url, "
            f"using {known_models[0]!r} instead", "WARN", True)
        defaults["model"] = known_models[0]
    else:
        raise ValueError('models.json defines no models under "model_url".')

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
# A user-provided provider list is honoured, but only for providers this
# onnxruntime build actually ships; a setting that was copy-pasted onto a
# machine without that EP used to fail at InferenceSession() time.
_configured_providers = config.get("settings", {}).get("ortProviders")
if _configured_providers:
    _unknown = [p for p in _configured_providers if p not in available_providers]
    if _unknown:
        log(f"Configured ORT providers are not available in this onnxruntime build and were "
            f"ignored: {', '.join(_unknown)}", "WARN", True)
    defaults["ortProviders"] = [p for p in _configured_providers
                                if p in available_providers and p not in _ORT_BLOCKLIST]
else:
    defaults["ortProviders"] = [p for p in _ORT_PRIORITY
                                if p in available_providers and p not in _ORT_BLOCKLIST]
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

log("Available ORT providers: " +
    ", ".join(onnxruntime.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " +
    ", ".join(defaults["ortProviders"]), "DEBUG", True)

def _migrate_legacy_model(model_name, dest_model, dest_meta):
    """Move legacy flat files into the nested directory structure (v1.x -> v2.x).

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
        legacy_preprocess = os.path.join(legacy_dir, model_name + ".preprocess.json")

        os.makedirs(os.path.dirname(dest_model), exist_ok=True)
        if not os.path.exists(dest_model):
            os.rename(legacy_model, dest_model)

        for legacy_meta in (legacy_csv, legacy_json):
            if os.path.exists(legacy_meta):
                os.makedirs(os.path.dirname(dest_meta), exist_ok=True)
                if not os.path.exists(dest_meta):
                    os.rename(legacy_meta, dest_meta)
                break

        # Migrate preprocess.json if present (animetimm models)
        preprocess_path = config.get("preprocess_path", {}).get(model_name)
        if preprocess_path and os.path.exists(legacy_preprocess):
            dest_preprocess = os.path.join(models_dir, preprocess_path)
            os.makedirs(os.path.dirname(dest_preprocess), exist_ok=True)
            if not os.path.exists(dest_preprocess):
                os.rename(legacy_preprocess, dest_preprocess)

        log(f"Migrated legacy files for {model_name} from {legacy_dir} to nested layout", "INFO", True)
        return


# ComfyUI hands us an IMAGE tensor (float32 [B, H, W, 3] in 0..1), so each family
# gets a `_prep_*` that stays on the tensor (GPU when it is on the GPU) plus the
# original `*_tag_batch` PIL path, which `settings.preprocess="pil"` selects.
# test/harness.py checks that both agree on real models.


def _to_nchw(images: torch.Tensor) -> torch.Tensor:
    """[B, H, W, 3] -> [B, 3, H, W] without copying when possible."""
    if images.dim() != 4:
        raise ValueError(f"expected a [B, H, W, C] image tensor, got {tuple(images.shape)}")
    if images.shape[1] == 3:
        return images
    if images.shape[-1] == 3:
        return images.permute(0, 3, 1, 2)
    raise ValueError(f"could not find the channel axis in {tuple(images.shape)}")


def _resize_shortest_side(images: torch.Tensor, target: int):
    """Replicate PIL's `ratio = target / max(size)` letterbox geometry."""
    h, w = images.shape[-2:]
    ratio = float(target) / max(h, w)
    return max(1, int(h * ratio)), max(1, int(w * ratio))


def _find_input_layout(shape):
    """Return (layout, height, width) for a 4-D ONNX input shape."""
    if len(shape) != 4:
        raise ValueError(f"expected a 4-D model input, got {shape}")
    if shape[1] == 3:                       # NCHW
        size = [shape[i] if isinstance(shape[i], int) else 448 for i in (2, 3)]
        return "NCHW", size[0], size[1]
    size = [shape[i] if isinstance(shape[i], int) else 448 for i in (1, 2)]   # NHWC
    return "NHWC", size[0], size[1]


def _prep_wd(images, height, width):
    """WD taggers: NHWC BGR float32 in 0..255 on a white canvas."""
    x = _to_nchw(images)
    nh, nw = _resize_shortest_side(x, height)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    x = F.pad(x, ((width - nw) // 2, width - nw - (width - nw) // 2,
                  (height - nh) // 2, height - nh - (height - nh) // 2), value=1.0)
    x = x.mul(255).flip(1).permute(0, 2, 3, 1).contiguous()  # RGB->BGR, NCHW->NHWC
    return x.detach().cpu().numpy()


def _prep_pixai(images, height, width):
    """Pixai: [B, 3, H, W] in -1..1 on a mid-grey canvas, dataloader-style."""
    x = _to_nchw(images)
    nh, nw = _resize_shortest_side(x, height)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    x = F.pad(x, ((width - nw) // 2, width - nw - (width - nw) // 2,
                  (height - nh) // 2, height - nh - (height - nh) // 2), value=128.0 / 255.0)
    return x.sub(0.5).div(0.5).detach().cpu().numpy()


def _prep_camie(images, height, width):
    """Camie: [B, 3, H, W] ImageNet-normalised on its signature grey canvas."""
    x = _to_nchw(images)
    nh, nw = _resize_shortest_side(x, height)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    # The canvas is per-channel (124, 116, 104), so pad each channel separately
    # to match Image.new("RGB", ..., (124, 116, 104)) exactly.
    pad = ((width - nw) // 2, width - nw - (width - nw) // 2,
           (height - nh) // 2, height - nh - (height - nh) // 2)  # l, r, t, b
    x = F.pad(x, pad, mode="constant", value=0.0)
    if any(pad):
        # torch.full, not expand(): expand() is a zero-stride view and cannot be
        # assigned into.
        fill = torch.tensor([124, 116, 104], dtype=x.dtype, device=x.device).div(255.0)
        canvas = fill.view(1, 3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[3]).contiguous()
        y0, x0 = pad[2], pad[0]
        canvas[:, :, y0:y0 + nh, x0:x0 + nw] = x[:, :, y0:y0 + nh, x0:x0 + nw]
        x = canvas
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return x.sub(mean).div(std).detach().cpu().numpy()


def _prep_cl_v2(images, height, width):
    """CL Tagger v2: plain RGB resize to a square, -1..1, no letterboxing."""
    x = F.interpolate(_to_nchw(images), size=(height, width),
                      mode="bicubic", align_corners=False, antialias=True)
    return x.sub(0.5).div(0.5).detach().cpu().numpy()


def _prep_cl_v1(images, target_size, is_nchw):
    """CL Tagger v1: square white letterbox, RGB, -1..1 (BGR when NCHW)."""
    x = _to_nchw(images)
    b, _, h, w = x.shape
    side = max(h, w)
    x = F.pad(x, ((side - w) // 2, side - w - (side - w) // 2,
                  (side - h) // 2, side - h - (side - h) // 2), value=1.0)
    x = F.interpolate(x, size=(target_size, target_size),
                      mode="bicubic", align_corners=False, antialias=True)
    if is_nchw:
        x = x.flip(1)
    else:
        x = x.permute(0, 2, 3, 1).contiguous()
    return x.sub(0.5).div(0.5).detach().cpu().numpy()


def _prep_animetimm(images, pad_size, resize_size, crop_size, mean, std):
    """AnimeTimm: white-pad to pad_size, resize, centre crop, then normalise."""
    x = _to_nchw(images)
    b, _, h, w = x.shape
    if h < pad_size[0] or w < pad_size[1]:
        x = F.pad(x, (0, max(0, pad_size[1] - w), 0, max(0, pad_size[0] - h)), value=1.0)
    x = F.interpolate(x, size=tuple(resize_size), mode="bicubic",
                      align_corners=False, antialias=True)
    # torchvision's CenterCrop rounds the crop origin up; match it exactly.
    ch, cw = crop_size
    top = int(round((x.shape[2] - ch) / 2.0))
    left = int(round((x.shape[3] - cw) / 2.0))
    x = x[:, :, top:top + ch, left:left + cw]
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return x.sub(mean).div(std).detach().cpu().numpy()


def wd_tag_batch(wd_model: InferenceSession, images: list[Image.Image]):
    """Run WD tagger on a batch of PIL images in a single ONNX call."""
    img_input = wd_model.get_inputs()[0]
    (_, height, width, channel) = img_input.shape

    batch = []
    for img in images:
        ratio = float(height) / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (height, height), (255, 255, 255))
        paste_x = (height - new_size[0]) // 2
        paste_y = (height - new_size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        img_np = np.array(new_img, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        batch.append(img_np)

    batch_np = np.stack(batch, axis=0)  # [B, H, W, C]
    label_name = wd_model.get_outputs()[0].name
    return wd_model.run([label_name], {img_input.name: batch_np})[0]


def pixai_tag_batch(pixai_model: InferenceSession, images: list[Image.Image]):
    """Run Pixai tagger on a batch of PIL images in a single ONNX call."""
    img_input = pixai_model.get_inputs()[0]
    (_, channel, height, width) = img_input.shape

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch = []
    for img in images:
        ratio = float(height) / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (height, height), (128, 128, 128))
        paste_x = (height - new_size[0]) // 2
        paste_y = (height - new_size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        batch.append(transform(new_img))

    batch_t = torch.stack(batch, dim=0).numpy()  # [B, C, H, W]
    pred_name = pixai_model.get_outputs()[2].name
    return pixai_model.run([pred_name], {img_input.name: batch_t})[0]


def animetimm_tag_batch(animetimm_model: InferenceSession, images: list[Image.Image], preprocess: dict):
    """Run animetimm tagger on a batch of PIL images in a single ONNX call."""
    steps = preprocess["test"]
    pad_step = next(s for s in steps if s["type"] == "pad_to_size")
    resize_step = next(s for s in steps if s["type"] == "resize")
    crop_step = next(s for s in steps if s["type"] == "center_crop")
    norm_step = next(s for s in steps if s["type"] == "normalize")

    pad_size = tuple(pad_step["size"])
    resize_size = resize_step["size"]
    crop_size = crop_step["size"]
    mean = norm_step["mean"]
    std = norm_step["std"]

    transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    batch = []
    for img in images:
        img_np = np.array(img.convert("RGB"))
        h, w = img_np.shape[:2]
        if h < pad_size[0] or w < pad_size[1]:
            new_h = max(h, pad_size[0])
            new_w = max(w, pad_size[1])
            padded = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
            off_h = (new_h - h) // 2
            off_w = (new_w - w) // 2
            padded[off_h:off_h + h, off_w:off_w + w] = img_np
            img = Image.fromarray(padded)
        batch.append(transform(img))

    batch_t = torch.stack(batch, dim=0).numpy()  # [B, C, H, W]
    img_input = animetimm_model.get_inputs()[0]
    output_names = [o.name for o in animetimm_model.get_outputs()]
    outputs = animetimm_model.run(output_names, {img_input.name: batch_t})
    best_idx = max(range(len(outputs)), key=lambda i: outputs[i].shape[-1])
    logits = outputs[best_idx]
    return 1.0 / (1.0 + np.exp(-logits))


def camie_tag_batch(camie_model: InferenceSession, images: list[Image.Image]):
    """Run Camie tagger on a batch of PIL images in a single ONNX call."""
    img_input = camie_model.get_inputs()[0]
    (_, channel, height, width) = img_input.shape

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch = []
    for img in images:
        ratio = float(height) / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (height, height), (124, 116, 104))
        paste_x = (height - new_size[0]) // 2
        paste_y = (height - new_size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        batch.append(transform(new_img))

    batch_t = torch.stack(batch, dim=0).numpy()  # [B, C, H, W]
    init_pred_name = camie_model.get_outputs()[0].name
    refine_pred_name = camie_model.get_outputs()[1].name
    select_cand_name = camie_model.get_outputs()[2].name
    (_, ref_logits, _) = camie_model.run(
        [init_pred_name, refine_pred_name, select_cand_name], {img_input.name: batch_t})
    return 1.0 / (1.0 + np.exp(-ref_logits))


def cl_tagger_v2_tag_batch(cl_model: InferenceSession, images: list[Image.Image]):
    """Run CL Tagger v2 on a batch of PIL images in a single ONNX call."""
    img_input = cl_model.get_inputs()[0]
    (_, channel, height, width) = img_input.shape

    batch = []
    for img in images:
        img = img.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        img_np = img_np.transpose(2, 0, 1)  # [C, H, W]
        batch.append(img_np)

    batch_np = np.stack(batch, axis=0)  # [B, C, H, W]
    logits_name = cl_model.get_outputs()[0].name
    logits = cl_model.run([logits_name], {img_input.name: batch_np})[0]
    return 1.0 / (1.0 + np.exp(-logits))


def _load_animetimm_preprocess(model_name: str) -> dict:
    """Load the per-model preprocess.json from its model subdirectory."""
    path = os.path.join(models_dir, config["preprocess_path"][model_name])
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


def cl_tagger_v1_tag_batch(cl_model: InferenceSession, images: list[Image.Image]):
    """Run CL Tagger v1 on a batch of PIL images in a single ONNX call."""
    img_input = cl_model.get_inputs()[0]
    input_shape = img_input.shape

    # Detect layout: NCHW = [B, 3, H, W], NHWC = [B, H, W, 3]
    layout, size_h, size_w = _find_input_layout(input_shape)
    is_nchw = layout == "NCHW"
    target_size = size_h if is_nchw else size_w

    batch = []
    for img in images:
        w, h = img.size
        if w != h:
            new_s = max(w, h)
            new_img = Image.new("RGB", (new_s, new_s), (255, 255, 255))
            new_img.paste(img, ((new_s - w) // 2, (new_s - h) // 2))
            img = new_img
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = img_np[:, :, ::-1]  # RGB -> BGR
        batch.append(img_np)

    batch_np = np.stack(batch, axis=0)  # [B, H, W, C]
    if is_nchw:
        batch_np = batch_np.transpose(0, 3, 1, 2)  # [B, C, H, W]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    else:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    batch_np = (batch_np - mean) / std
    logits_name = cl_model.get_outputs()[0].name
    logits = cl_model.run([logits_name], {img_input.name: batch_np})[0]
    return 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------
# Per-model binding: layout, preprocessing and precomputed tag-selection data.
#
# Everything here is derived once per session. The hot loop then only does
# numpy indexing and string joins, with no per-image pandas work.
# ---------------------------------------------------------------------------

class ModelSpec:
    """A loaded session bound to its preprocessing and precomputed tag tables.

    Built once per execution. The per-image work is then pure numpy indexing, so
    no DataFrame is rebuilt or re-scanned for each image.
    """

    def __init__(self, sess, df, model_name, preprocess, prep, run_mode):
        self.sess = sess
        self.model_name = model_name
        self.preprocess = preprocess
        self.prep = prep              # callable(_to_nchw(tensor)) -> model input array
        self.run_mode = run_mode      # "plain" | "camie" | "animetimm" | "sigmoid"
        img_input = sess.get_inputs()[0]
        self.in_name = img_input.name
        self.out_names = [o.name for o in sess.get_outputs()]

        categories = df["category"].to_numpy() if "category" in df.columns \
            else np.zeros(len(df), dtype=np.int64)
        self.tags = {
            name: np.flatnonzero(categories == cat)
            for name, cat in (("general", 0), ("character", 4), ("rating", 1))
        }
        raw = df["name"].to_numpy(dtype=object)
        self.raw_names = np.asarray(raw, dtype=object)
        # Pre-escape so _format_tags never rebuilds the same strings per image.
        self.escaped_names = np.asarray(
            [str(n).replace("(", "\\(").replace(")", "\\)") for n in raw], dtype=object)
        self.best_threshold = np.nan_to_num(
            df["best_threshold"].to_numpy(dtype=np.float32), nan=1.0, posinf=1.0, neginf=1.0) \
            if "best_threshold" in df.columns else None

    def prepare_batch(self, images) -> np.ndarray:
        return self._run(self.prep(_to_nchw(images)))

    def prepare_pil_batch(self, images) -> np.ndarray:
        if self.model_name.startswith("animetimm"):
            return animetimm_tag_batch(self.sess, images, self.preprocess)
        if self.model_name.startswith("pixai"):
            return pixai_tag_batch(self.sess, images)
        if self.model_name.startswith("camie"):
            return camie_tag_batch(self.sess, images)
        if self.model_name.startswith("cl-tagger-v2"):
            return cl_tagger_v2_tag_batch(self.sess, images)
        if self.model_name.startswith("cl-tagger-v1"):
            return cl_tagger_v1_tag_batch(self.sess, images)
        return wd_tag_batch(self.sess, images)

    def _run(self, arr):
        outs = self.sess.run(self.out_names, {self.in_name: arr})
        if self.run_mode == "camie":
            return outs[1]                      # the "refine" head
        if self.run_mode == "animetimm":
            widest = max(range(len(outs)), key=lambda i: outs[i].shape[-1])
            logits = outs[widest]
        else:
            logits = outs[0]
        return 1.0 / (1.0 + np.exp(-logits)) if self.run_mode in ("animetimm", "sigmoid") else logits


def _build_spec(sess: InferenceSession, tagger_info) -> ModelSpec:
    """Bind a session to its family's preprocessing and tag tables."""
    df, model_name = tagger_info[0], tagger_info[1]
    preprocess = tagger_info[2] if len(tagger_info) > 2 else None
    layout, height, width = _find_input_layout(sess.get_inputs()[0].shape)

    if model_name.startswith("animetimm"):
        steps = preprocess["test"]
        pad = tuple(next(s for s in steps if s["type"] == "pad_to_size")["size"])
        resize = next(s for s in steps if s["type"] == "resize")["size"]
        crop = next(s for s in steps if s["type"] == "center_crop")["size"]
        norm = next(s for s in steps if s["type"] == "normalize")
        return ModelSpec(sess, df, model_name, preprocess,
                         lambda x: _prep_animetimm(x, pad, resize, crop, norm["mean"], norm["std"]),
                         "animetimm")
    if model_name.startswith("camie"):
        return ModelSpec(sess, df, model_name, preprocess,
                         lambda x: _prep_camie(x, height, width), "camie")
    if model_name.startswith("pixai"):
        return ModelSpec(sess, df, model_name, preprocess,
                         lambda x: _prep_pixai(x, height, width), "plain")
    if model_name.startswith("cl-tagger-v2"):
        return ModelSpec(sess, df, model_name, preprocess,
                         lambda x: _prep_cl_v2(x, height, height), "sigmoid")
    if model_name.startswith("cl-tagger-v1"):
        return ModelSpec(sess, df, model_name, preprocess,
                         lambda x: _prep_cl_v1(x, height, layout == "NCHW"), "sigmoid")
    return ModelSpec(sess, df, model_name, preprocess,
                     lambda x: _prep_wd(x, height, width), "plain")


def _escape(tag: str) -> str:
    return tag.replace("(", "\\(").replace(")", "\\)")


def _format_tags(tag_list, trailing_comma=False):
    if not tag_list:
        return ""
    res = ("" if trailing_comma else ", ").join((tag + (", " if trailing_comma else "") for tag in tag_list))
    return res


def _pick_top_rating(df):
    """Pick the single highest-probability tag from rating category (1)."""
    rating_rows = df[(df['category'] == 1) & (df['probs'] > 0)]
    if rating_rows.empty:
        return ""
    top = rating_rows.loc[rating_rows['probs'].idxmax()]
    return top['name']


def _pick_top_rating_np(names, probs, idx_rating):
    """Same as _pick_top_rating, on precomputed indices."""
    if not len(idx_rating):
        return ""
    values = probs[idx_rating]
    pos = values > 0
    if not pos.any():
        return ""
    return names[idx_rating[np.argmax(np.where(pos, values, -np.inf))]]


# Category convention for model loaders:
#   0 = general (descriptive tags, shown in general_tags)
#   1 = rating  (4 tags: safe/sensitive/questionable/explicit, shown in rating output)
#   2 = quality (4 tags: best/normal/bad/worst, excluded from general/character)
#   3 = meta    (metadata / model info, excluded from general/character)
#   4 = character (character/copyright/artist, shown in character_tags)


def get_tag(probs, tags_df: pd.DataFrame, spec=None, threshold=0.35, character_threshold=0.85,
            use_best_threshold=True, trailing_comma=False, sort_tags=False, exclude_tags=""):
    """Select tags for one image.

    `spec` carries the indices/names precomputed at load time. It is optional so
    callers (and tests) can still pass a plain DataFrame and get the original
    pandas behaviour.
    """
    if spec is None:
        df = tags_df.assign(probs=probs)
        if sort_tags:
            df = df.sort_values(by='probs', ascending=False)
        if use_best_threshold and 'best_threshold' in df.columns:
            best = df['best_threshold'].fillna(1.0)
            general = df[(df['category'] == 0) & (df['probs'] >= np.maximum(best, threshold))]['name'].to_list()
            character = df[(df['category'] == 4) & (df['probs'] >= np.maximum(best, character_threshold))]['name'].to_list()
        else:
            general = df[(df['category'] == 0) & (df['probs'] > threshold)]['name'].to_list()
            character = df[(df['category'] == 4) & (df['probs'] > character_threshold)]['name'].to_list()
        rating = _pick_top_rating(df)
        general = [_escape(t) for t in general]
        character = [_escape(t) for t in character]
    else:
        idx_general = spec.tags["general"]
        idx_character = spec.tags["character"]
        if use_best_threshold and spec.best_threshold is not None:
            # NaN best_threshold becomes 1.0, so `np.maximum(best, user)` keeps
            # the user's threshold in charge exactly like `fillna(1.0)` did.
            general = idx_general[probs[idx_general] >= np.maximum(spec.best_threshold[idx_general], threshold)]
            character = idx_character[probs[idx_character] >= np.maximum(spec.best_threshold[idx_character], character_threshold)]
        else:
            general = idx_general[probs[idx_general] >= threshold]
            character = idx_character[probs[idx_character] >= character_threshold]
        if sort_tags:
            # numpy's argsort is ascending; the pandas path used ascending=False.
            general = general[np.argsort(probs[general])[::-1]]
            character = character[np.argsort(probs[character])[::-1]]
        rating = _pick_top_rating_np(spec.raw_names, probs, spec.tags["rating"])
        general = spec.escaped_names[general].tolist()
        character = spec.escaped_names[character].tolist()

    remove = [s.strip() for s in exclude_tags.lower().split(",")] if exclude_tags else []
    if remove:
        def _apply_exclude(tag_list):
            # Escape for output, but match against the unescaped name.
            return [_escape(t) for t in tag_list
                    if t.replace("\\(", "(").replace("\\)", ")").lower() not in remove]
        character = _apply_exclude(character)
        general = _apply_exclude(general)

    tags = character + general

    return {
        "combined": _format_tags(tags, trailing_comma),
        "rating": rating,
        "character": _format_tags(character, trailing_comma),
        "general": _format_tags(general, trailing_comma),
    }


class _DownloadProgress:
    """Aggregate download progress of all model files onto one node progress bar."""

    def __init__(self):
        self._bar = None      # comfy.utils.ProgressBar
        self._completed = 0   # bytes fully downloaded from previous files

    def __call__(self, total, downloaded):
        # Invoked after headers (downloaded == 0) and then for every chunk.
        if total <= 0:
            return  # Server sent no Content-Length; the console tqdm still reports progress.
        if self._bar is None:
            self._bar = utils.ProgressBar(self._completed + total)
            self._bar.update_absolute(0)  # Make the bar appear immediately.
            return
        self._bar.update_absolute(
            min(self._completed + downloaded, self._completed + total),
            total=self._completed + total)
        if downloaded >= total:
            self._completed += total


async def download_model(model: str) -> None:
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    hf_endpoint = hf_endpoint.rstrip("/")

    url = config["model_url"][model].replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main"

    model_path = config["model_path"].get(model, model + ".onnx")
    metadata_path = config["metadata_path"].get(model, "selected_tags.csv")
    external_data_path = config.get("external_data_path", {}).get(model)
    dest_model_path = os.path.join(models_dir, model_path)
    dest_metadata_path = os.path.join(models_dir, metadata_path)

    # Remote paths default to the basename of the local ones; they only need to be
    # listed in models.json when the remote layout differs from the local one.
    remote_model_path = config.get("remote_model_path", {}).get(model, os.path.basename(model_path))
    remote_metadata_path = config.get("remote_metadata_path", {}).get(model, os.path.basename(metadata_path))
    remote_external_data_path = config.get("remote_external_data_path", {}).get(model, external_data_path)
    preprocess_url = config.get("preprocess_url", {}).get(model, config["model_url"][model])
    preprocess_url = f"{preprocess_url.replace('{HF_ENDPOINT}', hf_endpoint).rstrip('/')}/resolve/main"

    # Support HF token for gated models.
    # Priority: HF_TOKEN env var -> HUGGINGFACE_TOKEN env var -> huggingface_hub cache (hf auth login)
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

    progress = _DownloadProgress()
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            # Only download the ONNX model file if it does not exist locally
            if not os.path.exists(dest_model_path):
                os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)
                log(f"Downloading model {model} to {dest_model_path}...", "INFO", True)
                await download_to_file(f"{url}/{remote_model_path}", dest_model_path,
                                       session=session, progress_cb=progress)

            # Only download external data file if required and missing
            if external_data_path:
                dest_ext_path = os.path.join(models_dir, external_data_path)
                if not os.path.exists(dest_ext_path):
                    os.makedirs(os.path.dirname(dest_ext_path), exist_ok=True)
                    log(f"Downloading external data file for {model} to {dest_ext_path}...", "INFO", True)
                    await download_to_file(f"{url}/{remote_external_data_path}", dest_ext_path,
                                           session=session, progress_cb=progress)

            # Only download the metadata file if it does not exist locally
            if not os.path.exists(dest_metadata_path):
                os.makedirs(os.path.dirname(dest_metadata_path), exist_ok=True)
                log(f"Downloading metadata to {dest_metadata_path}...", "INFO", True)
                await download_to_file(f"{url}/{remote_metadata_path}", dest_metadata_path,
                                       session=session, progress_cb=progress)

            # Only download preprocess.json if required (animetimm models)
            preprocess_path = config.get("preprocess_path", {}).get(model)
            if preprocess_path:
                dest_preprocess_path = os.path.join(models_dir, preprocess_path)
                if not os.path.exists(dest_preprocess_path):
                    os.makedirs(os.path.dirname(dest_preprocess_path), exist_ok=True)
                    log(f"Downloading preprocess for {model} to {dest_preprocess_path}...", "INFO", True)
                    await download_to_file(f"{preprocess_url}/preprocess.json", dest_preprocess_path,
                                           session=session, progress_cb=progress)

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
                io.Boolean.Input("use_best_threshold", default=True,
                                 tooltip="Use AnimeTimm's per-tag best_threshold values as minimum thresholds."),
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
    def execute(cls, tagger_model, tagger_info, image, threshold, character_threshold,
                use_best_threshold=True, trailing_comma=False, sort_tags=False, exclude_tags="") -> io.NodeOutput:
        tags_df = tagger_info[0]
        model_name = tagger_info[1]
        # AnimeTimm preprocessing is loaded alongside its model metadata. Reuse
        # it here instead of rereading preprocess.json for every execution.
        preprocess = tagger_info[2] if model_name.startswith("animetimm") else None

        # Bind layout + precomputed tag indices once per execution instead of
        # re-deriving them for every image.
        spec = _build_spec(tagger_model, (tags_df, model_name, preprocess))

        # Models with a fixed batch of 1 must be run one image at a time; models
        # with a dynamic batch dim get the whole list in a single ONNX call.
        batch_dim = tagger_model.get_inputs()[0].shape[0]
        can_batch = not isinstance(batch_dim, int) or batch_dim != 1
        total = image.shape[0]
        chunk = total if can_batch else 1

        pbar = utils.ProgressBar(total)
        probs = np.empty((total, len(tags_df)), dtype=np.float32)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            if _PREPROCESS == "pil":
                images = [Image.fromarray(np.asarray(image[i].mul(255).clamp(0, 255)
                                                     .to(torch.uint8).cpu())) for i in range(start, stop)]
                rows = spec.prepare_pil_batch(images)
            else:
                rows = spec.prepare_batch(image[start:stop])
            probs[start:stop] = rows
            pbar.update(stop - start)

        tags_list, ratings_list, chara_list, general_list = [], [], [], []
        for i in range(total):
            result = get_tag(probs[i], tags_df, spec, threshold,
                             character_threshold, use_best_threshold, trailing_comma, sort_tags, exclude_tags)
            tags_list.append(result["combined"])
            ratings_list.append(result["rating"])
            chara_list.append(result["character"])
            general_list.append(result["general"])
        return io.NodeOutput(tags_list, general_list, ratings_list, chara_list)


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
    async def execute(cls, model_name, replace_underscore) -> io.NodeOutput:
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
        external_data_path = config.get("external_data_path", {}).get(model_name)
        if external_data_path and not os.path.exists(os.path.join(models_dir, external_data_path)):
            needs_download = True
        preprocess_path = config.get("preprocess_path", {}).get(model_name)
        if preprocess_path and not os.path.exists(os.path.join(models_dir, preprocess_path)):
            needs_download = True
        if needs_download:
            await download_model(model_name)

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
            # Remap WD rating tags from category 9 -> 1 (rating)
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
