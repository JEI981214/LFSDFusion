# Pretrained weights

LFSDFusion does not ship third-party weights. Download them from their official
sources and place them under `pretrained/` as shown below. Please comply with
the license of each model.

| Model | Version / file | Official source | License |
|---|---|---|---|
| Segment Anything | ViT-B, `sam_vit_b_01ec64.pth` (358 MB) | https://github.com/facebookresearch/segment-anything (checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | Apache-2.0 |
| Stable Diffusion | 2.1-base (VAE + U-Net + CLIP text encoder + tokenizer), HF diffusers layout | https://huggingface.co/stabilityai/stable-diffusion-2-1-base | CreativeML OpenRAIL-M |
| LLaVA | v1.5-7B, HuggingFace transformers format | https://huggingface.co/llava-hf/llava-1.5-7b-hf | Llama-2 community license (request access on HF) |
| LPIPS | VGG for training loss; AlexNet for evaluation | auto-downloaded by the `lpips` package | BSD-style (see lpips repo) |

Expected local layout:

```
pretrained/
├── stable-diffusion-2-1-base/     # diffusers layout: unet/ vae/ text_encoder/ tokenizer/
├── sam_vit_b_01ec64.pth
└── llava-1.5-7b-hf/               # HF transformers layout (config, safetensors, tokenizer, processor)
```

In each `configs/*.yaml`, the paths are set to `./pretrained/...`; adjust
`paths.sd_dir`, `paths.sam_ckpt`, `paths.llava_dir` if you keep the models
elsewhere.

Only LoRA adapters (r=4) on the SD VAE encoder and U-Net are trained; the SAM,
SD, CLIP, and LLaVA backbones are kept frozen.
