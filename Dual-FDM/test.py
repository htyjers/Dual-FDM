import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL

import os
base_model_path = "/RealVisXL_V5.0"
image_encoder_path = "/image_encoder"
ip_ckpt = "/ip-adapter-plus_sdxl_vit-h.bin"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    # print(w, h)
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    add_watermarker=False,
)

device = "cuda:0"
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)


i = 0

line = 'robot + A photograph of a robot helping an elderly woman cross the street + /object/64.jpg + /Realism/05.jpg + 568-_object_64-_Realism_052.jpg'
parts = [part.strip() for part in line.split('+')]
line = line.strip()
if line:
    parts = [part.strip() for part in line.split('+')]
    if i >= 0:
        ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        image_content = Image.open(f"/path/{parts[2]}")
        image_content = image_content.resize((512, 512))
        image_style = Image.open(f"/path/4.jpg")
        image_style = image_style.resize((512, 512))
        num_samples = 1
        images = ip_model.generate(pil_image_content=image_content, pil_image_style=image_style,
                                   num_samples=num_samples, num_inference_steps=30, seed=42,
                                   prompt=parts[1], prompt_content=parts[0], scale=0.6)
        grid = image_grid(images, 1, num_samples)
        os.makedirs(f"/path", exist_ok=True)#1_c_baseline_s_our
        grid.save(f"/path/{parts[4]}")
    i += 1
