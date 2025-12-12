from .pysssss import init
from .nodes import WD14TaggerExtension

WEB_DIRECTORY = "./web"

async def comfy_entrypoint() -> WD14TaggerExtension:
    if init(check_imports=["onnxruntime"]):
        return WD14TaggerExtension()
    else:
        exit(1)

