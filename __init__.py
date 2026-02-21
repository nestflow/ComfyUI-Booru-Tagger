from .utils import init
from .nodes import BooruTaggerExtension

# WEB_DIRECTORY = "./web"

async def comfy_entrypoint() -> BooruTaggerExtension:
    if init(check_imports=["onnxruntime"]):
        return BooruTaggerExtension()
    else:
        exit(1)

