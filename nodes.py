from comfy_api.latest import ComfyExtension, io
import numpy as np
import asyncio
import os
import aiohttp
import folder_paths
import sys
import onnxruntime
from server import PromptServer
from aiohttp import web
from PIL import Image
from .pysssss import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config, log
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

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
else:
    models_dir = get_ext_dir("models", mkdir=True)
known_models = list(config["model_url"].keys())

log("Available ORT providers: " +
    ", ".join(onnxruntime.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " +
    ", ".join(defaults["ortProviders"]), "DEBUG", True)


def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    # models = [m for m in models if os.path.exists(
    #     os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models


async def wd_tag(wd_model: InferenceSession, img: Image.Image):
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


async def pixai_tag(pixai_model: InferenceSession, img):
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


async def camie_tag(camie_model: InferenceSession, img):
    img_input = camie_model.get_inputs()[0]
    (batch_size, channel, height, width) = img_input.shape

    # Reduce to max size and pad with white
    ratio = float(height)/max(img.size)
    new_size = tuple([int(x*ratio) for x in img.size])

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


def get_tag(probs, tags_df: pd.DataFrame, threshold=0.35, character_threshold=0.85, trailing_comma=False, sort_tags=False, exclude_tags=""):
    df = tags_df.assign(probs=probs)
    if sort_tags:
        df = df.sort_values(by='probs', ascending=False)

    general = df[(df['category'] == 0) & (df['probs'] > threshold)]['name'].to_list()
    character = df[(df['category'] == 4) & (df['probs'] > character_threshold)]['name'].to_list()
    tags = character + general

    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    tags = [tag for tag in tags if tag not in remove]

    res = ("" if trailing_comma else ", ").join((tag.replace(
        "(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for tag in tags))

    return res


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["model_url"][model]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main"

    model_path = config["model_path"].get(model, "model.onnx")
    metadata_path = config["metadata_path"].get(model, "selected_tags.csv")

    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}/{model_path}", os.path.join(models_dir, f"{model}.onnx"), update_callback, session=session)

            ext = metadata_path.split('.')[-1]
            await download_to_file(
                f"{url}/{metadata_path}", os.path.join(models_dir, f"{model}.{ext}"), update_callback, session=session)

        except aiohttp.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

        update_node_status(client_id, node, None)

    return web.Response(status=200)


class WD14Tagger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WD14Tagger",
            category="image",
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
            ]
        )

    @classmethod
    def execute(cls, tagger_model, tagger_info, image, threshold, character_threshold, trailing_comma=False, sort_tags=False, exclude_tags="") -> io.NodeOutput:
        pbar = utils.ProgressBar(image.shape[0])
        tags = []
        for i in range(image.shape[0]):
            img = Image.fromarray(np.array(image[i] * 255, dtype=np.uint8))

            tags_df = tagger_info[0]
            model_name = tagger_info[1]

            if model_name.startswith("pixai-tagger"):
                probs = wait_for_async(lambda: pixai_tag(tagger_model, img))
            elif model_name.startswith("camie-tagger-v2"):
                probs = wait_for_async(lambda: camie_tag(tagger_model, img))
            else:  # WD tagger
                probs = wait_for_async(lambda: wd_tag(tagger_model, img))

            tags.append(get_tag(probs, tags_df, threshold,
                        character_threshold, trailing_comma, sort_tags, exclude_tags))
            pbar.update(1)
        return io.NodeOutput(tags)


class LoadTaggerModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        extra = [name for name, _ in (os.path.splitext(
            m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return io.Schema(
            node_id="LoadTaggerModel",
            category="model",
            inputs=[
                io.Combo.Input("model_name", options=models,
                               default=defaults["model"]),
                io.Boolean.Input("replace_underscore",
                                 default=defaults["replace_underscore"]),
            ],
            outputs=[
                io.Custom("TAGGER_MODEL").Output("tagger_model"),
                io.Custom("TAGGER_INFO").Output("tagger_info")
            ]
        )

    @classmethod
    async def execute(cls, model_name, replace_underscore, client_id=None, node=None) -> io.NodeOutput:
        # Load model
        if model_name.endswith(".onnx"):
            model_name = model_name[0:-5]
        installed = list(get_installed_models())
        if not any(model_name + ".onnx" in s for s in installed):
            await download_model(model_name, client_id, node)

        name = os.path.join(models_dir, model_name + ".onnx")
        model = InferenceSession(name, providers=defaults["ortProviders"])

        csv_path = os.path.join(models_dir, model_name + ".csv")
        json_path = os.path.join(models_dir, model_name + ".json")
        if (model_name.startswith("wd") or model_name.startswith("pixai")) and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            return io.NodeOutput(model, (df, model_name))
        elif model_name.startswith("camie") and os.path.exists(json_path):
            df = pd.DataFrame()
            with open(json_path) as f:
                js = json.load(f)
                tag_mapping = js["dataset_info"]["tag_mapping"]
                df["name"] = list(tag_mapping["idx_to_tag"].values())
                df["category_name"] = list(
                    tag_mapping["tag_to_category"].values())
                category_to_idx = {value: index for index, value in enumerate(
                    js["dataset_info"]["categories"])}
                df["category"] = df["category_name"].replace(category_to_idx)
            if replace_underscore:
                df["name"] = df["name"].str.replace("_", " ")
            return io.NodeOutput(model, (df, model_name))
        else:
            log("No tag data is found.")
            exit(1)


class UniqueTags(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UniqueTags",
            category="text",
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


class WD14TaggerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadTaggerModel,
            WD14Tagger,
            UniqueTags
        ]
