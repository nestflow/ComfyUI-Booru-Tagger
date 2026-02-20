from comfy_api.latest import ComfyExtension, io
import numpy as np
import csv
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
known_models = list(config["models"].keys())

log("Available ORT providers: " +
    ", ".join(onnxruntime.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " +
    ", ".join(defaults["ortProviders"]), "DEBUG", True)


def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(
        os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models


async def tag(wd14_model, image):
    img_input = wd14_model.get_inputs()[0]

    model_type = 'pixai' if img_input.shape[1] == 3 else 'wd'

    if model_type == 'pixai':
        (batch_size, channel, height, width) = img_input.shape
    else:
        (batch_size, height, width, channel) = img_input.shape

    # Reduce to max size and pad with white
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR

    if model_type == 'pixai':
        image = np.transpose(image, (2, 0, 1))
    
    image = np.expand_dims(image, 0) # Batch dim
    
    if model_type == 'pixai':
        logit_name = wd14_model.get_outputs()[0].name
        pred_name = wd14_model.get_outputs()[1].name
        (logits, prediction) = wd14_model.run([logit_name, pred_name], {img_input.name: image})
        print(logits)
        print(prediction)
        result = logits[0]
    else:
        label_name = wd14_model.get_outputs()[0].name
        probs = wd14_model.run([label_name], {img_input.name: image})[0]
        result = probs[0]
    
    return result


def get_tag(probs, wd14_tag_info: pd.DataFrame, threshold=0.35, character_threshold=0.85, trailing_comma=False, exclude_tags=""):
    df = wd14_tag_info.copy()
    df['probs'] = probs

    general = df[(df['category'] == 0) & (df['probs'] > threshold)]['name'].to_list()
    character = df[(df['category'] == 4) & (df['probs'] > character_threshold)]['name'].to_list()
    tags = character + general

    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    tags = [tag for tag in tags if tag not in remove]

    res = ("" if trailing_comma else ", ").join((tag.replace(
        "(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for tag in tags))
    
    print('Tags:', res)
    return res


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["models"][model]
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}model.onnx", os.path.join(models_dir, f"{model}.onnx"), update_callback, session=session)
            await download_to_file(
                f"{url}selected_tags.csv", os.path.join(models_dir, f"{model}.csv"), update_callback, session=session)
        except aiohttp.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

        update_node_status(client_id, node, None)

    return web.Response(status=200)


@PromptServer.instance.routes.get("/pysssss/wd14tagger/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(type)
    image_path = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))

    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image = Image.open(image_path)

    models = get_installed_models()
    default = defaults["model"] + ".onnx"
    model = default if default in models else models[0]

    return web.json_response(await tag(image, model, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


class WD14Tagger(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WD14Tagger",
            category="image",
            inputs=[
                io.Custom("WD14_MODEL").Input("wd14_model"),
                io.Custom("WD14_TAG_INFO").Input("wd14_tag_info"),
                io.Image.Input("image"),
                io.Float.Input("threshold", min=0.0, max=1.0,
                               step=0.05, default=defaults["threshold"]),
                io.Float.Input("character_threshold",
                               min=0.0, max=1.0, step=0.05, default=defaults["character_threshold"]),
                io.Boolean.Input("trailing_comma",
                                 default=defaults["trailing_comma"]),
                io.String.Input(
                    "exclude_tags", default=defaults["exclude_tags"], multiline=True),
            ],
            outputs=[
                io.String.Output("tags", is_output_list=True),
            ]
        )

    @classmethod
    def execute(cls, wd14_model, wd14_tag_info, image, threshold, character_threshold, trailing_comma=False, exclude_tags="") -> io.NodeOutput:
        pbar = utils.ProgressBar(image.shape[0])
        tags = []
        for i in range(image.shape[0]):
            img = Image.fromarray(np.array(image[i] * 255, dtype=np.uint8))
            probs = wait_for_async(lambda: tag(wd14_model, img))

            tags.append(get_tag(probs, wd14_tag_info, threshold, character_threshold, trailing_comma, exclude_tags))
            pbar.update(1)
        return io.NodeOutput(tags)

class LoadWD14Model(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        extra = [name for name, _ in (os.path.splitext(
            m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return io.Schema(
            node_id="LoadWD14Model",
            category="model",
            inputs=[
                io.Combo.Input("model_name", options=models, default=defaults["model"]),
                io.Boolean.Input("replace_underscore", default=defaults["replace_underscore"]),
            ],
            outputs=[
                io.Custom("WD14_MODEL").Output("wd14_model"),
                io.Custom("WD14_TAG_INFO").Output("wd14_tag_info")
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

        # Read all tags from csv and locate start of each category
        df = pd.read_csv(os.path.join(models_dir, model_name + ".csv"))
        if replace_underscore:
            df["name"] = df["name"].str.replace("_", " ")

        # tags = []
        # general_index = None
        # character_index = None
        # with open(os.path.join(models_dir, model_name + ".csv")) as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         if general_index is None and row[2] == "0":
        #             general_index = reader.line_num - 2
        #         elif character_index is None and row[2] == "4":
        #             character_index = reader.line_num - 2
        #         if replace_underscore:
        #             tags.append(row[1].replace("_", " "))
        #         else:
        #             tags.append(row[1])
        
        return io.NodeOutput(model, df)

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
            LoadWD14Model,
            WD14Tagger,
            UniqueTags
        ]
