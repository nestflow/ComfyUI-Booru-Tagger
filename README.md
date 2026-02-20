# ComfyUI Booru Tagger

## Modification from [pythongosssss](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

1. Migrate to ComfyUI Node v3.
2. Separate model loading and inference, much faster running! (No longer need to load models for each image input).
3. Add support for [Pixai Tagger v0.9 (onnx model)](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) and [Camie Tagger v2](https://huggingface.co/Camais03/camie-tagger-v2).

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension allowing the interrogation of booru tags from images.

Credits:
- [pythongosssss/ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)
- [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags)
- [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)


Models created by 
- WD Taggers: [SmilingWolf](https://huggingface.co/SmilingWolf).
- Pixai Tagger: [pixai-labs](https://huggingface.co/pixai-labs).
- Camie Tagger: [Camais03](https://huggingface.co/Camais03).

## Installation
1. Clone this repo into the `custom_nodes` folder.
2. Install dependency (`onnxruntime` or `onnxruntime-gpu`).
