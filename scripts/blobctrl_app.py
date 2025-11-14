##!/usr/bin/python3
# -*- coding: utf-8 -*-
import gradio as gr
import os, sys
import json
import copy

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.utils import save_image
from transformers import AutoImageProcessor, Dinov2Model
from segment_anything import SamPredictor, sam_model_registry

from diffusers import (
    UNet2DConditionModel, 
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler,
)
from huggingface_hub import snapshot_download


from blobctrl.utils.utils import splat_features, viz_score_fn, BLOB_VIS_COLORS, vis_gt_ellipse_from_ellipse
from blobctrl.pipelines.pipeline_blobnet import StableDiffusionBlobNetPipeline
from blobctrl.models.blobnet import BlobNetModel


weight_dtype = torch.float16
device = "cuda"

# download blobctrl models
BlobCtrl_path = "models"
if not (os.path.exists(f"{BlobCtrl_path}/blobnet") and os.path.exists(f"{BlobCtrl_path}/unet_lora")):
    BlobCtrl_path = snapshot_download(
        repo_id="Yw22/BlobCtrl",
        local_dir=BlobCtrl_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"BlobCtrl checkpoints downloaded to {BlobCtrl_path}")

# download stable-diffusion-v1-5
StableDiffusion_path = "models/stable-diffusion-v1-5"
if not os.path.exists(StableDiffusion_path):
    StableDiffusion_path = snapshot_download(
        repo_id="sd-legacy/stable-diffusion-v1-5",
        local_dir=StableDiffusion_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"StableDiffusion checkpoints downloaded to {StableDiffusion_path}")

# download dinov2-large
Dino_path = "models/dinov2-large"
if not os.path.exists(Dino_path):
    Dino_path = snapshot_download(
        repo_id="facebook/dinov2-large",
        local_dir=Dino_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"Dino checkpoints downloaded to {Dino_path}")

# download SAM model
SAM_path = "models/sam/sam_vit_h_4b8939.pth"
if not os.path.exists(SAM_path):
    os.makedirs(os.path.dirname(SAM_path), exist_ok=True)
    import urllib.request
    print(f"Downloading SAM model...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        SAM_path
    )
    print(f"SAM model downloaded to {SAM_path}")


## load models and pipeline
blobnet_path = "./models/blobnet"
unet_lora_path = "./models/unet_lora"
stabel_diffusion_model_path = "./models/stable-diffusion-v1-5"
dinov2_path = "./models/dinov2-large"
sam_path = "./models/sam/sam_vit_h_4b8939.pth"

## unet
print(f"Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
       stabel_diffusion_model_path, 
       subfolder="unet", 
)
with torch.no_grad():
    initial_input_channels = unet.config.in_channels
    new_conv_in = torch.nn.Conv2d(
        initial_input_channels + 1,
        unet.conv_in.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=unet.conv_in.bias is not None,
        dtype=unet.dtype,
        device=unet.device,
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :initial_input_channels].copy_(unet.conv_in.weight)
    if unet.conv_in.bias is not None:
        new_conv_in.bias.copy_(unet.conv_in.bias)
    unet.conv_in = new_conv_in

## blobnet
print(f"Loading BlobNet...")
blobnet = BlobNetModel.from_pretrained(blobnet_path, ignore_mismatched_sizes=True)

## sam
print(f"Loading SAM...")
mobile_sam = sam_model_registry['vit_h'](checkpoint=sam_path).to(device)
mobile_sam.eval()
mobile_predictor = SamPredictor(mobile_sam)
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]
rgba_colors = [(255, 0, 255, 255), (0, 255, 0, 255), (0, 0, 255, 255)]

## dinov2
print(f"Loading Dinov2...")
dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_path)
dinov2 = Dinov2Model.from_pretrained(dinov2_path).to(device)


## stable diffusion with blobnet pipeline
print(f"Loading StableDiffusionBlobNetPipeline...")
pipeline = StableDiffusionBlobNetPipeline.from_pretrained(
        stabel_diffusion_model_path,
        unet=unet,
        blobnet=blobnet,
        torch_dtype=weight_dtype,
        dinov2_processor=dinov2_processor,
        dinov2=dinov2,
)

print(f"Loading UNetLora...")
pipeline.load_lora_weights(
    unet_lora_path,
    adapter_name="default",
)
pipeline.set_adapters(["default"])

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
# pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.to(device)
pipeline.set_progress_bar_config(leave=False)


## meta info
logo = r"""
<center><img src='././assets/logo_512.png' alt='BlobCtrl logo' style="width:80px; margin-bottom:10px"></center>
"""


head= r"""
<div style="text-align: center;">
    <h1> BlobCtrl: Taming Controllable Blob for Element-level Image Editing </h1>
    <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href='https://liyaowei-stu.github.io/project/BlobCtrl/'><img src='https://img.shields.io/badge/Project_Page-BlobCtrl-green' alt='Project Page'></a>
        <a href='http://arxiv.org/abs/2503.13434'><img src='https://img.shields.io/badge/Paper-Arxiv-blue'></a>
        <a href='https://github.com/TencentARC/BlobCtrl'><img src='https://img.shields.io/badge/Code-Github-orange'></a>
    </div>
    </br>
</div>
"""

descriptions = r"""
Official Gradio Demo for <a href=''><b>BlobCtrl: Taming Controllable Blob for Element-level Image Editing</b></a><br>
🦉 BlobCtrl enables precise, user-friendly element-level visual manipulation. <br>
"""


citation = r"""
If BlobCtrl is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/BlobCtrl' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC/BlobCtrl?style=social)]()
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@article{li2025blobctrl,
  title={BlobCtrl: Taming Controllable Blob for Element-level Image Editing},
  author={Li, Yaowei and Li, Lingen and Zhang, Zhaoyang and Li, Xiaoyu and Wang, Guangzhi and Li, Hongxiang and Cun, Xiaodong and Shan, Ying and Zou, Yuexian},
  journal={arXiv preprint arXiv:2503.13434},
  year={2025}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>liyaowei@gmail.com</b>.
"""

# - - - - - examples  - - - - -  #
EXAMPLES= [
                [
                "./assets/results/demo/move_hat/input_image/input_image.png",
                "A frog sits on a rock in a pond, with a top hat beside it, surrounded by butterflies and vibrant flowers.", 
                1.0,
                0.0,
                0.9,
                1248464818,
                0,
                ],
                [
                "./assets/results/demo/move_cup/input_image/input_image.png",
                "a rustic wooden table.", 
                1.0,
                0.0,
                1.0,
                1248464818,
                1,
                ],
                [
                "./assets/results/demo/enlarge_deer/input_image/input_image.png",
                "A cute, young deer with large ears standing in a grassy field at sunrise, surrounded by trees.",
                1.6,
                0.0,
                1.0,
                1288911487,
                2,
                ],
                [
                "./assets/results/demo/shrink_dragon/input_image/input_image.png",
                "A detailed, handcrafted cardboard dragon with red wings and expressive eyes.", 
                1.0,
                0.0,
                1.0,
                1248464818,
                3,
                ],
                [
                "./assets/results/demo/remove_shit/input_image/input_image.png",
                "The background consists of a textured, gray concrete surface with a red brick wall behind it. The bricks are arranged in a classic pattern, showcasing various shades of red and some weathering.", 
                1.0,
                0.0,
                1.0,
                1248464818,
                4,
                ],
                [
                "./assets/results/demo/remove_cow/input_image/input_image.png",
                "A majestic mountain range with rugged peaks under a cloudy sky, and a grassy field in the foreground.", 
                1.0,
                0.0,
                1.0,
                1248464818,
                5,
                ],
                [
                "./assets/results/demo/compose_rabbit/input_image/input_image.png",
                "A cute brown rabbit sitting on a wooden surface with a serene lake and mountains in the background.", 
                1.0,
                0.0,
                1.0,
                1248464818,
                6,
                ],
                [
                "./assets/results/demo/compose_cake/input_image/input_image.png",
                " slice of cake on a light blue background.", 
                1.2,
                0.0,
                1.0,
                1248464818,
                7,
                ],
                [
                "./assets/results/demo/replace_knife/input_image/input_image.png",
                "A slice of cake on a light blue background, with a knife in the center.",
                1.2,
                0.0,
                1.0,
                1248464818,
                8,
                ]
    ]
#
OBJECT_IMAGE_GALLERY = [
    ["./assets/results/demo/move_hat/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/move_cup/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/enlarge_deer/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/shrink_dragon/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/remove_shit/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/remove_cow/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/compose_rabbit/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/compose_cake/object_image_gallery/validation_object_region_center.png"],
    ["./assets/results/demo/replace_knife/object_image_gallery/validation_object_region_center.png"],
]
ORI_RESULT_GALLERY = [
    ["./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/move_cup/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/move_cup/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/move_cup/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/move_cup/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/move_cup/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/enlarge_deer/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/enlarge_deer/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/enlarge_deer/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/enlarge_deer/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/enlarge_deer/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/shrink_dragon/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/shrink_dragon/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/shrink_dragon/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/shrink_dragon/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/shrink_dragon/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/remove_shit/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/remove_shit/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/remove_shit/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/remove_shit/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/remove_shit/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/remove_cow/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/remove_cow/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/remove_cow/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/remove_cow/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/remove_cow/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/compose_rabbit/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/compose_rabbit/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/compose_rabbit/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/compose_rabbit/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/compose_rabbit/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/compose_cake/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/compose_cake/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/compose_cake/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/compose_cake/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/compose_cake/ori_result_gallery/ori_result_gallery_4.png"],
    ["./assets/results/demo/replace_knife/ori_result_gallery/ori_result_gallery_0.png", "./assets/results/demo/replace_knife/ori_result_gallery/ori_result_gallery_1.png", "./assets/results/demo/replace_knife/ori_result_gallery/ori_result_gallery_2.png", "./assets/results/demo/replace_knife/ori_result_gallery/ori_result_gallery_3.png", "./assets/results/demo/replace_knife/ori_result_gallery/ori_result_gallery_4.png"],
]
EDITABLE_BLOB = [
    "./assets/results/demo/move_hat/editable_blob/editable_blob.png",
    "./assets/results/demo/move_cup/editable_blob/editable_blob.png",
    "./assets/results/demo/enlarge_deer/editable_blob/editable_blob.png",
    "./assets/results/demo/shrink_dragon/editable_blob/editable_blob.png",
    "./assets/results/demo/remove_shit/editable_blob/editable_blob.png",
    "./assets/results/demo/remove_cow/editable_blob/editable_blob.png",
    "./assets/results/demo/compose_rabbit/editable_blob/editable_blob.png",
    "./assets/results/demo/compose_cake/editable_blob/editable_blob.png",
    "./assets/results/demo/replace_knife/editable_blob/editable_blob.png",
]
EDITED_RESULT_GALLERY = [
    ["./assets/results/demo/move_hat/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/move_hat/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/move_cup/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/move_cup/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/enlarge_deer/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/enlarge_deer/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/shrink_dragon/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/shrink_dragon/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/remove_shit/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/remove_shit/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/remove_cow/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/remove_cow/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/compose_rabbit/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/compose_rabbit/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/compose_cake/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/compose_cake/edited_result_gallery/edited_result_gallery_1.png"],
    ["./assets/results/demo/replace_knife/edited_result_gallery/edited_result_gallery_0.png", "./assets/results/demo/replace_knife/edited_result_gallery/edited_result_gallery_1.png"],
]
RESULTS_GALLERY = [
    ["./assets/results/demo/move_hat/results_gallery/results_gallery_0.png", "./assets/results/demo/move_hat/results_gallery/results_gallery_1.png", "./assets/results/demo/move_hat/results_gallery/results_gallery_2.png", "./assets/results/demo/move_hat/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/move_cup/results_gallery/results_gallery_0.png", "./assets/results/demo/move_cup/results_gallery/results_gallery_1.png", "./assets/results/demo/move_cup/results_gallery/results_gallery_2.png", "./assets/results/demo/move_cup/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/enlarge_deer/results_gallery/results_gallery_0.png", "./assets/results/demo/enlarge_deer/results_gallery/results_gallery_1.png", "./assets/results/demo/enlarge_deer/results_gallery/results_gallery_2.png", "./assets/results/demo/enlarge_deer/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/shrink_dragon/results_gallery/results_gallery_0.png", "./assets/results/demo/shrink_dragon/results_gallery/results_gallery_1.png", "./assets/results/demo/shrink_dragon/results_gallery/results_gallery_2.png", "./assets/results/demo/shrink_dragon/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/remove_shit/results_gallery/results_gallery_0.png", "./assets/results/demo/remove_shit/results_gallery/results_gallery_1.png", "./assets/results/demo/remove_shit/results_gallery/results_gallery_2.png", "./assets/results/demo/remove_shit/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/remove_cow/results_gallery/results_gallery_0.png", "./assets/results/demo/remove_cow/results_gallery/results_gallery_1.png", "./assets/results/demo/remove_cow/results_gallery/results_gallery_2.png", "./assets/results/demo/remove_cow/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/compose_rabbit/results_gallery/results_gallery_0.png", "./assets/results/demo/compose_rabbit/results_gallery/results_gallery_1.png", "./assets/results/demo/compose_rabbit/results_gallery/results_gallery_2.png", "./assets/results/demo/compose_rabbit/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/compose_cake/results_gallery/results_gallery_0.png", "./assets/results/demo/compose_cake/results_gallery/results_gallery_1.png", "./assets/results/demo/compose_cake/results_gallery/results_gallery_2.png", "./assets/results/demo/compose_cake/results_gallery/results_gallery_3.png"],
    ["./assets/results/demo/replace_knife/results_gallery/results_gallery_0.png", "./assets/results/demo/replace_knife/results_gallery/results_gallery_1.png", "./assets/results/demo/replace_knife/results_gallery/results_gallery_2.png", "./assets/results/demo/replace_knife/results_gallery/results_gallery_3.png"],
]
ELLIPSE_LISTS = [
    [[[[227.10665893554688, 118.85255432128906], [85.48122482299804, 103.65433502197266], 87.37393951416016], [1, 1, 1, 0], 0], [[[361.1066589355469, 367.85255432128906], [85.48122482299804, 103.65433502197266], 87.37393951416016], [1, 1, 1, 0], 1]],
    [[[[249.1703643798828, 149.63021850585938], [83.36424179077149, 115.79973449707032], 0.8257154226303101], [1, 1, 1, 0], 0], [[[245.1703643798828, 270.6302185058594], [83.36424179077149, 115.79973449707032], 0.8257154226303101], [1, 1, 1, 0], 1]],
    [[[[234.69358825683594, 255.60946655273438], [196.208619140625, 341.067111328125], 15.866915702819824], [1, 1, 1, 0], 0], [[[234.69358825683594, 255.60946655273438], [226.394560546875, 393.538974609375], 15.866915702819824], [1.2, 1, 1, 0], 2], [[[234.69358825683594, 255.60946655273438], [237.71428857421876, 413.21592333984376], 15.866915702819824], [1.05, 1, 1, 0], 2], [[[237.69358825683594, 237.60946655273438], [237.71428857421876, 413.21592333984376], 15.866915702819824], [1.05, 1, 1, 0], 1], [[[237.69358825683594, 233.60946655273438], [237.71428857421876, 413.21592333984376], 15.866915702819824], [1.05, 1, 1, 0], 1]],
    [[[[367.17742919921875, 201.1094512939453], [206.3889125696118, 377.8448820272314], 56.17562484741211], [1, 1, 1, 0], 0], [[[367.17742919921875, 201.1094512939453], [147.91468688964844, 297.0842980957031], 56.17562484741211], [0.8, 1, 1, 0], 2], [[[367.17742919921875, 201.1094512939453], [140.518952545166, 282.2300831909179], 56.17562484741211], [0.95, 1, 1, 0], 2], [[[324.17742919921875, 235.1094512939453], [140.518952545166, 282.2300831909179], 56.17562484741211], [0.95, 1, 1, 0], 1], [[[335.17742919921875, 225.1094512939453], [140.518952545166, 282.2300831909179], 56.17562484741211], [0.95, 1, 1, 0], 1]],
    [[[[255.23663330078125, 315.4020080566406], [263.64675201416014, 295.38494384765625], 153.8949432373047], [1, 1, 1, 0], 0]],
    [[[[335.09979248046875, 236.41409301757812], [168.37833966064454, 345.3470615478516], 0.7639619708061218], [1, 1, 1, 0], 0]],
    [[[[256.0, 256.0], [1e-05, 1e-05], 0], [1, 1, 1, 0], 0], [[[271.6672, 275.3536], [136.85061800371966, 303.75044578074284], 177.008], [1, 1, 1, 0], 0], [[[271.6672, 275.3536], [150.53567980409164, 303.75044578074284], 177.008], [1.1, 1, 1.1, 0], 4], [[[271.6672, 275.3536], [158.06246379429624, 318.93796806977997], 177.008], [1.05, 1, 1.1, 0], 2], [[[271.6672, 275.3536], [165.96558698401105, 334.88486647326897], 177.008], [1.05, 1, 1.1, 0], 2], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 177.008], [1.1, 1, 1.1, 0], 4], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 7.00800000000001], [1.1, 1, 1.1, 10], 5], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 3.0080000000000098], [1.1, 1, 1.1, -4], 5], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 177.008], [1.1, 1, 1.1, -6], 5], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 179.008], [1.1, 1, 1.1, 2], 5], [[[271.6672, 275.3536], [182.56214568241217, 334.88486647326897], 178.008], [1.1, 1, 1.1, -1], 5], [[[271.6672, 275.3536], [182.56214568241217, 368.3733531205959], 178.008], [1.1, 1.1, 1.1, -1], 3], [[[271.6672, 275.3536], [182.56214568241217, 349.95468546456607], 178.008], [1.1, 0.95, 1.1, -1], 3], [[[271.6672, 275.3536], [182.56214568241217, 349.95468546456607], 170.008], [1.1, 0.95, 1.1, -8], 5], [[[271.6672, 275.3536], [182.56214568241217, 349.95468546456607], 172.008], [1.1, 0.95, 1.1, 2], 5]],
    [[[[256.0, 256.0], [1e-05, 1e-05], 0], [1, 1, 1, 0], 0], [[[256.0, 256.0], [144.81546878700496, 144.81546878700496], 0], [1, 1, 1, 0], 0], [[[256.0, 256.0], [123.09314846895421, 123.09314846895421], 0], [0.85, 1, 1, 0], 2], [[[256.0, 256.0], [110.7838336220588, 110.7838336220588], 0], [0.9, 1, 1, 0], 2], [[[88.0, 418.0], [110.7838336220588, 110.7838336220588], 0], [0.9, 1, 1, 0], 1]],
    [[[[164.6718292236328, 385.8408508300781], [41.45796089172364, 319.87034912109374], 142.05267333984375], [1, 1, 1, 0], 0]],
]
TRACKING_POINTS = [
    [[227, 118], [361, 367]],
    [[249, 150], [248, 269]],
    [[234, 255], [234, 255], [234, 255], [237, 237], [237, 233]],
    [[367, 201], [367, 201], [367, 201], [324, 235], [335, 225]],
    [[255, 315]],
    [[335, 236]],
    [[256, 256], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271], [275, 271]],
    [[256, 256], [256, 256], [256, 256], [256, 256], [88, 418]],
    [[164, 385]],
]
REMOVE_STATE=[
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    False,
]
INPUT_IMAGE=[
    "./assets/results/demo/move_hat/input_image/input_image.png",
    "./assets/results/demo/move_cup/input_image/input_image.png",
    "./assets/results/demo/enlarge_deer/input_image/input_image.png",
    "./assets/results/demo/shrink_dragon/input_image/input_image.png",
    "./assets/results/demo/remove_shit/input_image/input_image.png",
    "./assets/results/demo/remove_cow/input_image/input_image.png",
    "./assets/results/demo/compose_rabbit/input_image/input_image.png",
    "./assets/results/demo/compose_cake/input_image/input_image.png",
    "./assets/results/demo/replace_knife/input_image/input_image.png",
]


## normal functions
def _get_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_copy = [contour_copy.tolist() for contour_copy in contours]
    
    concat_contours = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(concat_contours)
    ellipse = cv2.fitEllipse(hull)
    return ellipse, contours_copy


def ellipse_to_gaussian(x, y, a, b, theta):
    """
    Convert ellipse parameters to mean and covariance matrix of a Gaussian distribution.

    Parameters:
    x (float): x-coordinate of the ellipse center.
    y (float): y-coordinate of the ellipse center.
    a (float): Length of the minor semi-axis of the ellipse.
    b (float): Length of the major semi-axis of the ellipse.
    theta (float): Rotation angle of the ellipse (in radians), counterclockwise angle of the major axis.

    Returns:
    mean (numpy.ndarray): Mean of the Gaussian distribution, an array of shape (2,) representing (x, y) coordinates.
    cov_matrix (numpy.ndarray): Covariance matrix of the Gaussian distribution, an array of shape (2, 2).
    """
    # Mean
    mean = np.array([x, y])
    
    # Diagonal elements of the covariance matrix
    # sigma_x = b / np.sqrt(2)
    # sigma_y = a / np.sqrt(2)
    # Not dividing by sqrt(2) is also acceptable. This conversion is mainly for specific statistical contexts,
    # to make the semi-axis length of the ellipse correspond to one standard deviation of the Gaussian distribution.
    # The purpose is to make the ellipse area contain about 68% of the probability mass of the Gaussian distribution
    # (in a one-dimensional Gaussian distribution, one standard deviation contains about 68% of the probability mass).

    # Diagonal elements of the covariance matrix
    sigma_x = b 
    sigma_y = a 
    # Covariance matrix (before rotation)
    cov_matrix = np.array([[sigma_x**2, 0],
                            [0, sigma_y**2]])
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Rotate the covariance matrix
    cov_matrix_rotated = R @ cov_matrix @ R.T
    
    cov_matrix_rotated[0, 1] *= -1  # Reverse the non-diagonal elements of the covariance matrix
    cov_matrix_rotated[1, 0] *= -1  # Reverse the non-diagonal elements of the covariance matrix
    
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_rotated)
    
    return mean, cov_matrix_rotated

def normalize_gs(mean, cov_matrix_rotated, width, height):
    # Normalize mean
    normalized_mean = mean / np.array([width, height])
    
    # Calculate maximum length for normalizing the covariance matrix
    max_length = np.sqrt(width**2 + height**2)
    
    # Normalize covariance matrix
    normalized_cov_matrix = cov_matrix_rotated / (max_length ** 2)
    
    return normalized_mean, normalized_cov_matrix


def normalize_ellipse(ellipse, width, height):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse
    max_length = np.sqrt(width**2 + height**2)

    normalized_xc, normalized_yc = xc/width, yc/height
    normalized_d1, normalized_d2 = d1/max_length, d2/max_length
    return normalized_xc, normalized_yc, normalized_d1, normalized_d2, angle_clockwise_short_axis


def composite_mask_and_image(mask, image, masked_color=[0,0,0]):
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    if mask_np.ndim == 2:
        mask_indicator = (mask_np>0).astype(np.uint8)
    else:

        mask_indicator = (mask_np.sum(-1)>255).astype(np.uint8)
    masked_image = image_np * (1-mask_indicator[:,:,np.newaxis]) + masked_color * mask_indicator[:,:,np.newaxis]
    return Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")


def is_point_in_ellipse(point, ellipse):
    # 提取椭圆参数
    (xc, yc), (d1, d2), angle = ellipse
    
    # 将角度转换为弧度
    theta = np.radians(angle)
    
    # 计算相对坐标
    x, y = point
    x_prime = x - xc
    y_prime = y - yc
    
    # 计算旋转后的坐标
    x_rotated = x_prime * np.cos(theta) - y_prime * np.sin(theta)
    y_rotated = x_prime * np.sin(theta) + y_prime * np.cos(theta)
    
    # 计算椭圆方程，d1 和 d2 是全长轴和全短轴，需除以 2
    ellipse_equation = (x_rotated**2) / ((d1 / 2)**2) + (y_rotated**2) / ((d2 / 2)**2)
    
    # 判断点是否在椭圆内
    return ellipse_equation <= 1


def calculate_ellipse_vertices(ellipse):
    (xc, yc), (d1, d2), angle_clockwise_short_axis = ellipse


    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_clockwise_short_axis)
    
    # Calculate the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Calculate the unrotated vertices
  
    half_d1 = d1 / 2
    half_d2 = d2 / 2
    vertices = np.array([
        [half_d1, 0],  # Rightmost point on the long axis
        [-half_d1, 0],  # Leftmost point on the long axis
        [0, half_d2],  # Topmost point on the short axis
        [0, -half_d2]   # Bottommost point on the short axis
    ])

    # Rotate the vertices
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    
    # Translate vertices to the original center
    final_vertices = rotated_vertices + np.array([xc, yc])
    
    return final_vertices


def move_ellipse(ellipse, tracking_points):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse
    last_xc, last_yc = tracking_points[-1]
    second_last_xc, second_last_yc = tracking_points[-2]
    vx = last_xc - second_last_xc
    vy = last_yc - second_last_yc
    xc += vx
    yc += vy
    return (xc,yc), (d1,d2), angle_clockwise_short_axis


def resize_blob_func(ellipse, resizing_factor, height, width, resize_type):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse

    too_big = False
    too_small = False

    min_blob_area = 1600

    exceed_threshold = 0.4

    while True:
        if resize_type == 0:
            resized_d1 = d1 * resizing_factor
            resized_d2 = d2 * resizing_factor
        elif resize_type == 1:
            resized_d1 = d1 
            resized_d2 = d2 * resizing_factor
        elif resize_type == 2:
            resized_d1 = d1  * resizing_factor
            resized_d2 = d2
        resized_ellipse = (xc,yc), (resized_d1, resized_d2), angle_clockwise_short_axis
        resized_ellipse_vertices = calculate_ellipse_vertices(resized_ellipse)
        resized_ellipse_vertices = resized_ellipse_vertices / np.array([width, height])
        if resizing_factor != 1:
            # soft the threshold allowed to exceed the image range
            if np.all(resized_ellipse_vertices >= -exceed_threshold) and np.all(resized_ellipse_vertices <= 1+exceed_threshold):
                # calculate the blob area
                blob_area = np.pi * (resized_d1 / 2) * (resized_d2 / 2)
                if blob_area >= min_blob_area:
                    break
                else:
                    too_small = True
                    resizing_factor += 0.1
                    if blob_area < 1e-6:
                        ## if the blob area is too too too small, break
                        break
            else:
                too_big = True
                resizing_factor -= 0.1
        else:
            break

    if too_big:
        gr.Warning(f"The blob is too big, adaptive reduction of magnification to fit the image, The threshold allowed to exceed the image range is {exceed_threshold}")
    if too_small:
        gr.Warning(f"The blob is too small, adaptive enlargement of magnification to fit the image, The minimum blob area is {min_blob_area} px")
    return resized_ellipse, resizing_factor


def rotate_blob_func(ellipse, rotation_degree):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse
    rotated_angle_clockwise_short_axis = (angle_clockwise_short_axis + rotation_degree) % 180

    rotated_ellipse = (xc,yc), (d1,d2), rotated_angle_clockwise_short_axis
 
    return rotated_ellipse, rotation_degree


def get_theta_anti_clockwise_long_axis(angle_clockwise_short_axis):
    angle_anti_clockwise_short_axis = (180 - angle_clockwise_short_axis) % 180
    angle_anti_clockwise_long_axis = (angle_anti_clockwise_short_axis + 90) % 180
    theta_anti_clockwise_long_axis = np.radians(angle_anti_clockwise_long_axis)
    return theta_anti_clockwise_long_axis


def get_gs_from_ellipse(ellipse):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse
    theta_anti_clockwise_long_axis = get_theta_anti_clockwise_long_axis(angle_clockwise_short_axis)

    a = d1 / 2
    b = d2 / 2
    mean, cov_matrix = ellipse_to_gaussian(xc, yc, a, b, theta_anti_clockwise_long_axis)
    return mean, cov_matrix


def get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix):
    xs, ys = normalized_mean
    blob = {
        "xs": torch.tensor(xs).unsqueeze(0),
        "ys": torch.tensor(ys).unsqueeze(0),
        "covs":  torch.tensor(normalized_cov_matrix).unsqueeze(0).unsqueeze(0),
        "sizes": torch.tensor([1.0]).unsqueeze(0),
        }
    return blob


def clear_ellipse_lists(ellipse_lists):
    ellipse_lists = []
    return ellipse_lists


def get_blob_vis_img_from_blob_dict(blob, viz_size=64, score_size=64):
    blob_vis =  splat_features(**blob, 
                                interp_size=64, 
                                viz_size=viz_size,
                                is_viz=True, 
                                ret_layout=True, 
                                score_size=score_size,
                                viz_score_fn=viz_score_fn,
                                viz_colors=BLOB_VIS_COLORS,
                                only_vis=True)["feature_img"]
    blob_vis_img = blob_vis[0].permute(1,2,0).contiguous().cpu().numpy()
    blob_vis_img = (blob_vis_img*255).astype(np.uint8)
    blob_vis_img = Image.fromarray(blob_vis_img)
    return blob_vis_img


def get_blob_score_from_blob_dict(blob, score_size=64):
    blob_score = splat_features(**blob,
                                score_size=score_size,
                                return_d_score=True,
                                )[0]
    return blob_score


def get_object_region_from_mask(mask, original_image):
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask

    if mask_np.ndim == 2:
        mask_indicator = (mask_np>0).astype(np.uint8)
    else:
        mask_indicator = (mask_np.sum(-1)>255).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(mask_indicator)
    rect_mask = mask_indicator[y:y+h, x:x+w]

    tmp = original_image.copy()
    rect_region = tmp[y:y+h, x:x+w]

    rect_region_object_white_background = np.where(rect_mask[:, :, None] > 0, rect_region, 255)

    target_height, target_width = tmp.shape[:2]
    start_y = (target_height - h) // 2
    start_x = (target_width - w) // 2

    rect_region_object_white_background_center = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    rect_region_object_white_background_center[start_y:start_y+h, start_x:start_x+w] = rect_region_object_white_background
    rect_region_object_white_background_center = Image.fromarray(rect_region_object_white_background_center).convert("RGB")

    return rect_region_object_white_background_center


def extract_contours(object_image):
    """
    Extract contours from an object image
    :param object_image: Input object image, shape (h, w, 3), value range [0, 255]
    :return: Contour image
    """
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

    # 将图像二值化，假设物体不是白色 [255, 255, 255]
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    # 提取轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像用于绘制轮廓
    contour_image = np.zeros_like(gray_image)

    # 在空白图像上绘制轮廓
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)

    return contour_image


def get_mask_from_ellipse(ellipse, height, width):
    ellipse_mask_np = np.zeros((height, width))
    ellipse_mask_np = cv2.ellipse(ellipse_mask_np, ellipse, 255, -1)
    ellipse_mask = Image.fromarray(ellipse_mask_np).convert("L")
    return ellipse_mask


# gradio functions
@torch.no_grad()
def run_function(
                original_image,
                scene_prompt, 
                ori_result_gallery, 
                object_image_gallery,
                edited_result_gallery, 
                ellipse_lists, 
                blobnet_control_strength, 
                blobnet_control_guidance_start,
                blobnet_control_guidance_end,
                remove_blob_box,
                num_samples, 
                seed,
                guidance_scale,
                num_inference_steps,
                ## for save
                editable_blob,
                resize_blob_slider_maintain_aspect_ratio,
                resize_blob_slider_along_long_axis,
                resize_blob_slider_along_short_axis,
                rotation_blob_slider,
                resize_init_blob_slider,
                resize_init_blob_slider_long_axis,
                resize_init_blob_slider_short_axis,
                tracking_points,
                ):
    if object_image_gallery == [] or object_image_gallery == None or ori_result_gallery == [] or ori_result_gallery == None:
        gr.Warning("Please generate the blob first")
        return None
    
    if edited_result_gallery == [] or edited_result_gallery == None:
        gr.Warning("Please click the region in the blob in the first time.")
        return None

    generator = torch.Generator(device=device).manual_seed(seed)

    ## prepare img: object_region_center, edited_background_region
    gt_i_ellipse_img_path, masked_image_path, mask_image_path, ellipse_mask_path, ellipse_masked_image_path  = ori_result_gallery
    object_white_background_center_path = object_image_gallery[0]

    validation_object_region_center = Image.open(object_white_background_center_path[0])
    ori_ellipse_mask = Image.open(ellipse_mask_path[0])
    width, height = validation_object_region_center.size
    latent_height, latent_width = height // 8, width // 8


    if not remove_blob_box:
        edited_ellipse_masked_image_path, edited_ellipse_mask_path = edited_result_gallery
        validation_edited_background_region = Image.open(edited_ellipse_masked_image_path[0])
        ## prepare gs_score
        final_ellipse, final_transform_param, final_blob_edited_type = ellipse_lists[-1]
        mean, cov_matrix = get_gs_from_ellipse(final_ellipse)
        normalized_mean, normalized_cov_matrix = normalize_gs(mean, cov_matrix, width, height)
        blob_dict = get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix)   
        validation_gs_score = get_blob_score_from_blob_dict(blob_dict, score_size=(latent_height, latent_width)).unsqueeze(0).to(device) # bnhw
    else:
        img_tmp = original_image.copy()
        validation_edited_background_region = composite_mask_and_image(ori_ellipse_mask, img_tmp, masked_color=[255,255,255])
        ## prepare gs_score
        start_ellipse, start_transform_param, start_blob_edited_type = ellipse_lists[0]
        mean, cov_matrix = get_gs_from_ellipse(start_ellipse)
        normalized_mean, normalized_cov_matrix = normalize_gs(mean, cov_matrix, width, height)
        blob_dict = get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix)
        validation_gs_score = get_blob_score_from_blob_dict(blob_dict, score_size=(latent_height, latent_width)).unsqueeze(0).to(device) # bnhw
        validation_gs_score[:,0] = 1.0
        validation_gs_score[:,1] = 0.0
        final_ellipse = start_ellipse
        ## set blobnet control strength to 0.0
        blobnet_control_strength = 0.0

    with torch.autocast("cuda"):
        output = pipeline(
                fg_image=validation_object_region_center,
                bg_image=validation_edited_background_region,
                gs_score=validation_gs_score,
                generator=generator,
                prompt=[scene_prompt]*num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                blobnet_control_guidance_start=float(blobnet_control_guidance_start),
                blobnet_control_guidance_end=float(blobnet_control_guidance_end),
                blobnet_conditioning_scale=float(blobnet_control_strength),
                width=width,
                height=height,
                return_sample=False,
                )
        edited_images = output.images

    edit_image_plots = []
    for i in range(num_samples):
        edit_image_np = np.array(edited_images[i])
        edit_image_np_plot = cv2.ellipse(edit_image_np, final_ellipse, [0,255,0], 3)
        edit_image_plot = Image.fromarray(edit_image_np_plot).convert("RGB")
        edit_image_plots.append(edit_image_plot)

    results_gallery = [*edited_images, *edit_image_plots]

    ## save results
    # ori_save_path = "./assets/results/tmp/ori_result_gallery"
    # os.makedirs(ori_save_path, exist_ok=True)
    # # import ipdb; ipdb.set_trace()
    # for i in range(len(ori_result_gallery)):
    #     result = Image.open(ori_result_gallery[i][0])
    #     result.save(f"{ori_save_path}/ori_result_gallery_{i}.png")
 
    # object_save_path = "./assets/results/tmp/object_image_gallery"
    # os.makedirs(object_save_path, exist_ok=True)
    # validation_object_region_center.save(f"{object_save_path}/validation_object_region_center.png")
    
    # edited_save_path = "./assets/results/tmp/edited_result_gallery"
    # os.makedirs(edited_save_path, exist_ok=True)
    # for i in range(len(edited_result_gallery)):
    #     result = Image.open(edited_result_gallery[i][0])
    #     result.save(f"{edited_save_path}/edited_result_gallery_{i}.png")

    # results_save_path = "./assets/results/tmp/results_gallery"
    # os.makedirs(results_save_path, exist_ok=True)
    # for i in range(len(results_gallery)):
    #     results_gallery[i].save(f"{results_save_path}/results_gallery_{i}.png")

    # editable_blob_save_path = "./assets/results/tmp/editable_blob"
    # os.makedirs(editable_blob_save_path, exist_ok=True)
    # editable_blob_pil = Image.fromarray(editable_blob)
    # editable_blob_pil.save(f"{editable_blob_save_path}/editable_blob.png")

    # state_save_path = "./assets/results/tmp/state"
    # os.makedirs(state_save_path, exist_ok=True)
    # with open(f"{state_save_path}/state.json", "w") as f:
    #     json.dump({
    #         "blobnet_control_strength": blobnet_control_strength,
    #         "blobnet_control_guidance_start": blobnet_control_guidance_start,
    #         "blobnet_control_guidance_end": blobnet_control_guidance_end,
    #         "remove_blob_box": remove_blob_box,
    #         "num_samples": num_samples,
    #         "seed": seed,
    #         "guidance_scale": guidance_scale,
    #         "num_inference_steps": num_inference_steps,
    #         "ellipse_lists": ellipse_lists,
    #         "scene_prompt": scene_prompt,
    #         "resize_blob_slider_maintain_aspect_ratio": resize_blob_slider_maintain_aspect_ratio,
    #         "resize_blob_slider_along_long_axis": resize_blob_slider_along_long_axis,
    #         "resize_blob_slider_along_short_axis": resize_blob_slider_along_short_axis,
    #         "rotation_blob_slider": rotation_blob_slider,
    #         "resize_init_blob_slider": resize_init_blob_slider,
    #         "resize_init_blob_slider_long_axis": resize_init_blob_slider_long_axis,
    #         "resize_init_blob_slider_short_axis": resize_init_blob_slider_short_axis,
    #         "tracking_points": tracking_points,
    #     }, f)
    
    # input_image_save_path = "./assets/results/tmp/input_image"
    # os.makedirs(input_image_save_path, exist_ok=True)
    # Image.fromarray(original_image).save(f"{input_image_save_path}/input_image.png")

    torch.cuda.empty_cache()
    return results_gallery


def generate_blob(
    original_image, 
    original_mask, 
    selected_points, 
    ellipse_lists,
    init_resize_factor=1.05,
   ):
    if original_image is None:
        raise gr.Error('Please upload the input image') 

    if (original_mask is None) or (len(selected_points)==0):
        raise gr.Error("Please click the region where you hope unchanged/changed in input image to get segmentation mask")
    else:
        original_mask = np.clip(255 - original_mask, 0, 255).astype(np.uint8)


    ## get ellipse parameters from mask
    height, width = original_image.shape[:2]
    binary_mask = 255*(original_mask.sum(-1)>255).astype(np.uint8)
    ellipse, contours = _get_ellipse(binary_mask)
    ## properly enlarge ellipse to cover the whole blob
    ellipse, resizing_factor = resize_blob_func(ellipse, init_resize_factor, height, width, 0)

    ## get gaussian parameters from ellipse
    mean, cov_matrix = get_gs_from_ellipse(ellipse)
    normalized_mean, normalized_cov_matrix = normalize_gs(mean, cov_matrix, width, height)
    blob_dict = get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix)
    blob_vis_img =  get_blob_vis_img_from_blob_dict(blob_dict, viz_size=(height, width))
    
    ## plot masked image
    masked_image = composite_mask_and_image(original_mask, original_image)
    mask_image = Image.fromarray(original_mask.astype(np.uint8)).convert("L")

    ## get object region
    object_white_background_center = get_object_region_from_mask(original_mask, original_image)

    ## plot ellipse
    gt_i_ellipse = vis_gt_ellipse_from_ellipse(torch.tensor(original_image).round().contiguous().cpu().numpy(),
                                    ellipse,
                                    color=[0,255,0])
    gt_i_ellipse_img = Image.fromarray(gt_i_ellipse.astype(np.uint8))


    ellipse_mask = get_mask_from_ellipse(ellipse, height, width)
    ellipse_masked_image = composite_mask_and_image(ellipse_mask, original_image)

    ## return images
    ori_result_gallery = [gt_i_ellipse_img, masked_image, mask_image, ellipse_mask, ellipse_masked_image]
    object_image_gallery = [object_white_background_center]

    ## init ellipse_lists, 0: init, 1: move , 2: resize remain aspect ratio, 3: resize along long axis, 4: resize along short axis, 5: rotation
    ## ellipse_int = (ellipse, (resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle), blob_edited_type)
    ellipse_init = (ellipse, (1, 1, 1, 0), 0)
    if len(ellipse_lists) == 0:
        ellipse_lists.append(ellipse_init)
    else:
        ellipse_lists = clear_ellipse_lists(ellipse_lists)
        ellipse_lists.append(ellipse_init)

    ## init parameters
    rotation_blob_slider = 0
    resize_blob_slider_maintain_aspect_ratio = 1
    resize_blob_slider_along_long_axis = 1
    resize_blob_slider_along_short_axis = 1
    resize_init_blob_slider = 1
    resize_init_blob_slider_long_axis = 1
    resize_init_blob_slider_short_axis = 1
    init_ellipse_parameter = None
    init_object_image = None

    tracking_points = []
    edited_result_gallery = None

    return blob_vis_img, ori_result_gallery, object_image_gallery, ellipse_lists, tracking_points, edited_result_gallery, resize_blob_slider_maintain_aspect_ratio, resize_blob_slider_along_long_axis, resize_blob_slider_along_short_axis, rotation_blob_slider, resize_init_blob_slider, resize_init_blob_slider_long_axis, resize_init_blob_slider_short_axis, init_ellipse_parameter, init_object_image


# undo the selected point
def undo_seg_points(orig_img, sel_pix):
    # draw points
    output_mask = None
    if len(sel_pix) != 0:
        temp = orig_img.copy()
        sel_pix.pop()
        # online show seg mask
        if len(sel_pix) !=0:
            temp, output_mask = segmentation(temp, sel_pix)
        return temp.astype(np.uint8), output_mask
    else:
        gr.Warning("Nothing to Undo")


# once user upload an image, the original image is stored in `original_image`
def initialize_img(img):
    if max(img.shape[0], img.shape[1])*1.0/min(img.shape[0], img.shape[1])>2.0:
        raise gr.Error('image aspect ratio cannot be larger than 2.0')

    # Check if image needs resizing
    # Resize and crop to 512x512
    h, w = img.shape[:2]
    
    # First resize so shortest side is 512
    scale = 512 / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    
    # Then crop to 512x512
    h, w = img.shape[:2]
    start_y = (h - 512) // 2
    start_x = (w - 512) // 2
    img = img[start_y:start_y+512, start_x:start_x+512]

    original_image = img.copy()
    editable_blob = None
    selected_points = []
    tracking_points = []
    ellipse_lists = []
    ori_result_gallery = []
    object_image_gallery = []
    edited_result_gallery = []
    results_gallery = []
    blobnet_control_strength = 1.2
    blobnet_control_guidance_start = 0.0
    blobnet_control_guidance_end = 1.0
    resize_blob_slider_maintain_aspect_ratio = 1
    resize_blob_slider_along_long_axis = 1
    resize_blob_slider_along_short_axis = 1
    rotation_blob_slider = 0
    resize_init_blob_slider = 1
    resize_init_blob_slider_long_axis = 1
    resize_init_blob_slider_short_axis = 1
    init_ellipse_parameter = "[0.5, 0.5, 0.2, 0.2, 180]"
    init_object_image = None
    remove_blob_box = False 
    return img, original_image, editable_blob, selected_points, tracking_points, ellipse_lists, ori_result_gallery, object_image_gallery, edited_result_gallery, results_gallery, blobnet_control_strength, blobnet_control_guidance_start, blobnet_control_guidance_end, resize_blob_slider_maintain_aspect_ratio, resize_blob_slider_along_long_axis, resize_blob_slider_along_short_axis, rotation_blob_slider, resize_init_blob_slider, resize_init_blob_slider_long_axis, resize_init_blob_slider_short_axis, init_ellipse_parameter, init_object_image, remove_blob_box


# user click the image to get points, and show the points on the image
def segmentation(img, sel_pix):
    # online show seg mask
    points = []
    labels = []
    for p, l in sel_pix:
        points.append(p)
        labels.append(l)
    mobile_predictor.set_image(img if isinstance(img, np.ndarray) else np.array(img))
    with torch.no_grad():
        masks, _, _ = mobile_predictor.predict(point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False)

    output_mask = np.ones((masks.shape[1], masks.shape[2], 3))*255
    for i in range(3):
            output_mask[masks[0] == True, i] = 0.0

    mask_all = np.ones((masks.shape[1], masks.shape[2], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        mask_all[masks[0] == True, i] = color_mask[i]
    masked_img = img / 255 * 0.3 + mask_all * 0.7
    masked_img = masked_img*255
    ## draw points
    for point, label in sel_pix:
        cv2.drawMarker(masked_img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    return masked_img, output_mask


def get_point(img, sel_pix, evt: gr.SelectData):
    sel_pix.append((evt.index, 1))    # default foreground_point
    # online show seg mask
    masked_img, output_mask = segmentation(img, sel_pix)
    return masked_img.astype(np.uint8), output_mask


def tracking_points_for_blob(original_image, 
                            tracking_points, 
                            ellipse_lists, 
                            height, 
                            width, 
                            edit_status=True):

    sel_pix_transparent_layer = np.zeros((height, width, 4))
    sel_ell_transparent_layer = np.zeros((height, width, 4))

    start_ellipse, start_transform_param, start_blob_edited_type = ellipse_lists[0]
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[-1]

    ## plot start point
    start_point = tracking_points[0]
    cv2.drawMarker(sel_pix_transparent_layer, start_point, rgba_colors[-1], markerType=markers[1], markerSize=20, thickness=5)

    ## plot tracking points
    if len(tracking_points) > 1:
        tracking_points_real = []
        for point in tracking_points:
            if not tracking_points_real or point != tracking_points_real[-1]:
                tracking_points_real.append(point)

        for i in range(len(tracking_points_real)-1):
            start_point = tracking_points_real[i]
            end_point = tracking_points_real[i+1]
            vx = end_point[0] - start_point[0]
            vy = end_point[1] - start_point[1]
            arrow_length = np.sqrt(vx**2 + vy**2)

            ## draw arrow
            if i == len(tracking_points_real)-2:
                cv2.arrowedLine(sel_pix_transparent_layer, tuple(start_point), tuple(end_point), rgba_colors[-1], 2, tipLength=8 / arrow_length)
            else:
                cv2.line(sel_pix_transparent_layer, tuple(start_point), tuple(end_point), rgba_colors[-1], 2,)
        
        if edit_status:
            edited_ellipse = move_ellipse(current_ellipse, tracking_points_real)
            transform_param = current_transform_param
            ellipse_lists.append((edited_ellipse, transform_param, 1))

    ## draw ellipse, current ellipse need to be rearanged, because the ellipse_lists may be changed
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[-1]
    cv2.ellipse(sel_ell_transparent_layer, current_ellipse, rgba_colors[-1], 2, -1)

    # get current ellipse
    current_mean, current_cov_matrix = get_gs_from_ellipse(current_ellipse)
    current_normalized_mean, current_normalized_cov_matrix = normalize_gs(current_mean, current_cov_matrix, width, height)
    current_blob_dict = get_blob_dict_from_norm_gs(current_normalized_mean, current_normalized_cov_matrix)
    transparent_background = get_blob_vis_img_from_blob_dict(current_blob_dict, viz_size=(height, width)).convert('RGBA')

    ## composite images
    sel_pix_transparent_layer = Image.fromarray(sel_pix_transparent_layer.astype(np.uint8))
    sel_ell_transparent_layer = Image.fromarray(sel_ell_transparent_layer.astype(np.uint8))
    transform_gs_img = Image.alpha_composite(transparent_background, sel_pix_transparent_layer)
    transform_gs_img = Image.alpha_composite(transform_gs_img, sel_ell_transparent_layer)

    ## get vis edited image and mask
    # Use anti-aliasing to get smoother ellipse edges
    original_ellipse_mask_np = np.zeros((height, width), dtype=np.float32)
    original_ellipse_mask_np = cv2.ellipse(original_ellipse_mask_np, start_ellipse, 1.0, -1, lineType=cv2.LINE_AA)
    original_ellipse_mask_np = (original_ellipse_mask_np * 255).astype(np.uint8)
    original_ellipse_mask = Image.fromarray(original_ellipse_mask_np).convert("L")

    edited_ellipse_mask_np = np.zeros((height, width), dtype=np.float32) 
    edited_ellipse_mask_np = cv2.ellipse(edited_ellipse_mask_np, current_ellipse, 1.0, -1, lineType=cv2.LINE_AA)
    edited_ellipse_mask_np = (edited_ellipse_mask_np * 255).astype(np.uint8)
    edited_ellipse_mask = Image.fromarray(edited_ellipse_mask_np).convert("L")
    
    # import ipdb; ipdb.set_trace()

    original_ellipse_masked_image = composite_mask_and_image(original_ellipse_mask, original_image, masked_color=[255,255,255])
    edited_ellipse_masked_image = composite_mask_and_image(edited_ellipse_mask, original_ellipse_masked_image, masked_color=[0,0,0])
    edited_result_gallery = [edited_ellipse_masked_image, edited_ellipse_mask]

    return transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery


def add_tracking_points(original_image, 
                        tracking_points, 
                        ellipse_lists,
                        evt: gr.SelectData):  # SelectData is a subclass of EventData

    height, width = original_image.shape[:2]

    if len(ellipse_lists) == 0:
        gr.Warning("Please generate the blob first")
        return None, tracking_points, ellipse_lists, None

    ## get start ellipse
    start_ellipse, transform_param, blob_edited_type = ellipse_lists[0]
    ## check if the point is in the ellipse initially
    if not is_point_in_ellipse(evt.index, start_ellipse) and len(tracking_points) == 0:
        gr.Warning("Please click the region in the blob in the first time.")
        start_mean, start_cov_matrix = get_gs_from_ellipse(start_ellipse)
        start_normalized_mean, start_normalized_cov_matrix = normalize_gs(start_mean, start_cov_matrix, width, height)
        start_blob_dict = get_blob_dict_from_norm_gs(start_normalized_mean, start_normalized_cov_matrix)
        start_transparent_background = get_blob_vis_img_from_blob_dict(start_blob_dict, viz_size=(height, width)).convert('RGBA')
        return start_transparent_background, tracking_points, ellipse_lists, None

    if len(tracking_points) == 0:
        xc, yc = start_ellipse[0]
        tracking_points.append([int(xc), int(yc)])
    else:
        tracking_points.append(evt.index)


    tmp_img = original_image.copy()
    transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                        tracking_points, 
                                                                                                        ellipse_lists, 
                                                                                                        height, 
                                                                                                        width, 
                                                                                                        edit_status=True)
                                                                                                    



    return transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery


def undo_blob_points(original_image, tracking_points, ellipse_lists):
    height, width = original_image.shape[:2]
    if len(tracking_points) > 1:
        tmp_img = original_image.copy()
        tracking_points.pop()
        ellipse_lists.pop()
    
        transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                            tracking_points, 
                                                                                                            ellipse_lists, 
                                                                                                            height, 
                                                                                                            width, 
                                                                                                            edit_status=False)

        current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[-1]
        # resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle = current_transform_param
        resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle = 1,1,1,0

        return transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery, resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle
    else:
        if len(tracking_points) == 1:
            tracking_points.pop()
        else:
            gr.Warning("Nothing to Undo")
        transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery, resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle = reset_blob_points(original_image, tracking_points, ellipse_lists)
        return transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery, resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle


def reset_blob_points(original_image, tracking_points, ellipse_lists):
    edited_result_gallery = None
    height, width = original_image.shape[:2]
    tracking_points = []
    start_ellipse, start_transform_param, start_blob_edited_type = ellipse_lists[0]
    ellipse_lists = clear_ellipse_lists(ellipse_lists)
    ellipse_lists.append((start_ellipse, start_transform_param, start_blob_edited_type))
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[0]

    resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle = current_transform_param

    current_mean, current_cov_matrix = get_gs_from_ellipse(current_ellipse)
    current_normalized_mean, current_normalized_cov_matrix = normalize_gs(current_mean, current_cov_matrix, width, height)
    current_blob_dict = get_blob_dict_from_norm_gs(current_normalized_mean, current_normalized_cov_matrix)
    transform_gs_img = get_blob_vis_img_from_blob_dict(current_blob_dict, viz_size=(height, width)).convert('RGBA')
    return transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery, resizing_factor_remain_aspect_ratio, resizing_factor_long_axis, resizing_factor_short_axis, anti_clockwise_rotation_angle


def resize_blob(editable_blob, 
                original_image, 
                tracking_points, 
                ellipse_lists, 
                resizing_factor,
                resize_type,
                edited_result_gallery,
                remove_blob_box):
    if remove_blob_box:
        gr.Warning("Please use initial blob resize in remove mode to ensure the initial blob surrounds the object")
        return editable_blob, ellipse_lists, edited_result_gallery, 1


    if len(ellipse_lists) == 0:
        gr.Warning("Please generate the blob first")
        return None, ellipse_lists, None, 1
    if len(tracking_points) == 0:
        gr.Warning("Please select the blob first")
        return editable_blob, ellipse_lists, None, 1

    height, width = original_image.shape[:2]

    # resize_type: 0: maintain aspect ratio, 1: along long axis, 2: along short axis
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[-1]
    
    if resize_type == 0:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 0)
        transform_param = (resizing_factor, current_transform_param[1], current_transform_param[2], current_transform_param[3])
        ellipse_lists.append((edited_ellipse, transform_param, 2))
    elif resize_type == 1:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 1)
        transform_param = (current_transform_param[0], resizing_factor, current_transform_param[2], current_transform_param[3])
        ellipse_lists.append((edited_ellipse, transform_param, 3))
    elif resize_type == 2:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 2)
        transform_param = (resizing_factor, current_transform_param[1], resizing_factor, current_transform_param[3])
        ellipse_lists.append((edited_ellipse, transform_param, 4))


    ## reset resizing factor, resize is progressive
    resizing_factor = 1

    if len(tracking_points) > 0:
        tracking_points.append(tracking_points[-1])
    else:
        xc, yc = edited_ellipse[0]
        tracking_points.append([int(xc), int(yc)])

    tmp_img = original_image.copy()
    transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                        tracking_points, 
                                                                                                        ellipse_lists, 
                                                                                                        height, 
                                                                                                        width, 
                                                                                                        edit_status=False)
    
    return transform_gs_img, ellipse_lists, edited_result_gallery, resizing_factor


def resize_start_blob(editable_blob, 
                    original_image, 
                    tracking_points, 
                    ellipse_lists,
                    ori_result_gallery,
                    resizing_factor,
                    resize_type):
    if len(ellipse_lists) == 0:
        gr.Warning("Please generate the blob first")
        return None, ellipse_lists, None, None, 1
    if len(tracking_points) == 0:
        gr.Warning("Please select the blob first")
        return editable_blob, ellipse_lists, None, None, 1

    height, width = original_image.shape[:2]

    ## resize start blob for background
    current_idx = 0
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[current_idx]
    if resize_type == 0:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 0)
    elif resize_type == 1:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 1)
    elif resize_type == 2:
        edited_ellipse, resizing_factor = resize_blob_func(current_ellipse, resizing_factor, height, width, 2)
    transform_param = (current_transform_param[0], current_transform_param[1], current_transform_param[2], current_transform_param[3])
    ellipse_lists[0] = (edited_ellipse, transform_param, 0)
    ## reset resizing factor, resize along long axis is progressive
    resizing_factor = 1

    tmp_img = original_image.copy()
    transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                        tracking_points, 
                                                                                                        ellipse_lists, 
                                                                                                        height, 
                                                                                                        width, 
                                                                                                        edit_status=False)


    ## ori_result_gallery
    gt_i_ellipse_img_path, masked_image_path, mask_image_path, ellipse_mask_path, ellipse_masked_image_path  = ori_result_gallery
    masked_image = Image.open(masked_image_path[0])
    mask_image = Image.open(mask_image_path[0])

    ## new ellipse mask
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[current_idx]
    new_ellipse_mask_img = get_mask_from_ellipse(current_ellipse, height, width)
    new_ellipse_masked_img = composite_mask_and_image(new_ellipse_mask_img, tmp_img)

    gt_i_ellipse = vis_gt_ellipse_from_ellipse(torch.tensor(tmp_img).round().contiguous().cpu().numpy(),
                                                    current_ellipse,
                                                    color=[0,255,0])
    
    new_gt_i_ellipse_img = Image.fromarray(gt_i_ellipse.astype(np.uint8))

    ori_result_gallery = [new_gt_i_ellipse_img, masked_image, mask_image, new_ellipse_mask_img, new_ellipse_masked_img]
    
    return transform_gs_img, ellipse_lists, edited_result_gallery, ori_result_gallery, resizing_factor


def rotate_blob(editable_blob, 
                original_image, 
                tracking_points, 
                ellipse_lists, 
                rotation_degree):
    if len(ellipse_lists) == 0:
        gr.Warning("Please generate the blob first")
        return None, ellipse_lists, None, 0
    if len(tracking_points) == 0:
        gr.Warning("Please select the blob first")
        return editable_blob, ellipse_lists, None, 0

    height, width = original_image.shape[:2]
    current_idx = -1
    current_ellipse, current_transform_param, current_blob_edited_type = ellipse_lists[current_idx]
    edited_ellipse, rotation_degree = rotate_blob_func(current_ellipse, rotation_degree)
    transform_param = (current_transform_param[0], current_transform_param[1], current_transform_param[2], rotation_degree)
    ellipse_lists.append((edited_ellipse, transform_param, 5))
    rotation_degree = 0

    if len(tracking_points) > 0:
        tracking_points.append(tracking_points[-1])
    else:
        xc, yc = edited_ellipse[0]
        tracking_points.append([int(xc), int(yc)])

    tmp_img = original_image.copy()
    transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                        tracking_points, 
                                                                                                        ellipse_lists, 
                                                                                                        height, 
                                                                                                        width, 
                                                                                                        edit_status=False)
    return transform_gs_img, ellipse_lists, edited_result_gallery, rotation_degree


def remove_blob_box_func(editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, remove_blob_box):
    
    if remove_blob_box:
        return resize_start_blob(editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, 1.2, 0)
    else:
        return resize_start_blob(editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, 1.0, 0)


def set_init_ellipse(original_image, original_mask, edited_result_gallery, ellipse_lists, tracking_points, editable_blob, ori_result_gallery, init_ellipse_parameter):
    ## if init_ellipse_parameter is not None, use the manual initial ellipse
    if init_ellipse_parameter is not None and init_ellipse_parameter != "":
        # Parse string input like '[0.5,0.5,0.2,0.2,180]'
        params = eval(init_ellipse_parameter)
        normalized_xc, normalized_yc, normalized_d1, normalized_d2, angle_clockwise_short_axis = params
        height, width = original_image.shape[:2]
        max_length = np.sqrt(height**2 + width**2)
        ellipse_zero = ((width/2, height/2), (1e-5, 1e-5), 0)
        ellipse = ((normalized_xc*width, normalized_yc*height), (normalized_d1*max_length, normalized_d2*max_length), angle_clockwise_short_axis)
        original_mask = np.array(get_mask_from_ellipse(ellipse, height, width))
        original_mask = np.stack([original_mask, original_mask, original_mask], axis=-1)

        ellipse_init = (ellipse_zero, (1, 1, 1, 0), 0)
        ellipse_next = (ellipse, (1, 1, 1, 0), 0)

        if len(ellipse_lists) == 0:
            ellipse_lists.append(ellipse_init)
            ellipse_lists.append(ellipse_next)
        else:
            ellipse_lists = clear_ellipse_lists(ellipse_lists)
            ellipse_lists.append(ellipse_init)
            ellipse_lists.append(ellipse_next)


        tmp_img = original_image.copy()
        tracking_points = [[int(ellipse_init[0][0][1]), int(ellipse_init[0][0][0])], [int(ellipse_next[0][0][1]), int(ellipse_next[0][0][0])]]
        transform_gs_img, tracking_points, ellipse_lists, edited_result_gallery = tracking_points_for_blob(tmp_img, 
                                                                                                    tracking_points, 
                                                                                                    ellipse_lists, 
                                                                                                    height, 
                                                                                                    width, 
                                                                                                    edit_status=False)



        ## plot masked image
        masked_image = composite_mask_and_image(original_mask, original_image)
        mask_image = Image.fromarray(original_mask.astype(np.uint8)).convert("L")

        ## plot ellipse
        gt_i_ellipse = vis_gt_ellipse_from_ellipse(torch.tensor(original_image).round().contiguous().cpu().numpy(),
                                        ellipse,
                                        color=[0,255,0])
        gt_i_ellipse_img = Image.fromarray(gt_i_ellipse.astype(np.uint8))


        ellipse_mask = get_mask_from_ellipse(ellipse, height, width)
        ellipse_masked_image = composite_mask_and_image(ellipse_mask, original_image)
        ori_result_gallery = [gt_i_ellipse_img, masked_image, mask_image, ellipse_mask, ellipse_masked_image]
        return transform_gs_img, edited_result_gallery, ellipse_lists, tracking_points, ori_result_gallery, None

    gr.Warning("Please set the valid initial ellipse first")
    return editable_blob, edited_result_gallery, ellipse_lists, tracking_points, ori_result_gallery, "[0.5, 0.5, 0.2, 0.2, 180]"


def upload_object_image(object_image, edited_result_gallery, remove_blob_box):
    if edited_result_gallery == [] or edited_result_gallery == None:
        raise gr.Error("Please generate the blob first")
    else:
        # Check if image needs resizing
        # Resize and crop to 512x512
        h, w = object_image.shape[:2]
        
        # First resize so shortest side is 512
        scale = 512 / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        object_image = cv2.resize(object_image, (new_w, new_h))
        
        # Then crop to 512x512
        h, w = object_image.shape[:2]
        start_y = (h - 512) // 2
        start_x = (w - 512) // 2
        object_image = object_image[start_y:start_y+512, start_x:start_x+512]
        object_image_gallery = [object_image]
        remove_blob_box = False
        return object_image_gallery, remove_blob_box


block = gr.Blocks()
with block as demo:
    with gr.Row():
        with gr.Column(): 
            gr.HTML(head)

    gr.Markdown(descriptions)
    original_image = gr.State(value=None)
    original_mask = gr.State(value=None)

    resize_blob_maintain_aspect_ratio_state = gr.State(value=0)
    resize_blob_along_long_axis_state = gr.State(value=1)
    resize_blob_along_short_axis_state = gr.State(value=2)
    
    selected_points = gr.State([])
    tracking_points = gr.State([])
    ellipse_lists = gr.State([])

    with gr.Row():
        with gr.Column():
            with gr.Column(elem_id="Input"):
                gr.Markdown("## **Step 1: Upload an image and click to segment the object**", show_label=False)

                with gr.Row():
                    input_image = gr.Image(type="numpy", label="input", scale=2, height=576, interactive=True)
                    
                with gr.Row(elem_id="Seg"):
                    undo_seg_button = gr.Button('🔙 Undo Seg', elem_id="undo_btnSEG", scale=1)

                gr.Markdown("## **Step 2: Input the scene prompt and 🎩 generate the blob**", show_label=False)
                scene_prompt = gr.Textbox(label="Scene Prompt", value="Fill image using foreground and background.")
                generate_blob_button = gr.Button("🎩 Generate Blob",elem_id="btn")


                gr.Markdown("### 💡 Hint: Adjust the control strength and control timesteps range to balance appearance and flexibility", show_label=False)
                blobnet_control_strength = gr.Slider(label="🎚️ Control Strength:", minimum=0, maximum=2.5, value=1.6, step=0.01)

                with gr.Row():
                    blobnet_control_guidance_start = gr.Slider(label="Blobnet Control Timestep Start", minimum=0, maximum=1, step=0.01, value=0)
                    blobnet_control_guidance_end = gr.Slider(label="Blobnet Control Timestep End", minimum=0, maximum=1, step=0.01, value=0.9)

                gr.Markdown("### Click to adjust the diffusion sampling options 👇", show_label=False)
                with gr.Accordion("Diffusion Options", open=False, elem_id="accordion1"):                      
                    seed = gr.Slider(
                        label="Seed: ", minimum=0, maximum=2147483647, step=1, value=1248464818, scale=2
                    )

                    num_samples = gr.Slider(
                        label="Num samples", minimum=0, maximum=4, step=1, value=2
                    )

                    with gr.Group():
                        with gr.Row():
                            guidance_scale = gr.Slider(label="CFG scale", minimum=1, maximum=12, step=0.1, value=7.5)
                            num_inference_steps = gr.Slider(label="NFE", minimum=1, maximum=100, step=1, value=50)

            
        with gr.Column():
            gr.Markdown("### Click to expand more previews 👇", show_label=False)
            with gr.Row():
                with gr.Accordion("More Previews", open=False, elem_id="accordion2"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Tab(elem_classes="feedback", label="Object Image"):
                                object_image_gallery = gr.Gallery(label='Object Image', height=320, elem_id="gallery", show_label=True, interactive=False, preview=True)
                        with gr.Column():
                            with gr.Tab(elem_classes="feedback", label="Original Preview"):
                                ori_result_gallery = gr.Gallery(label='Original Preview', height=320, elem_id="gallery", show_label=True, interactive=False, preview=True)


            gr.Markdown("## **Step 3: Edit the blob, such as move/resize/remove the blob**", show_label=False)
            with gr.Row():
                with gr.Column():
                    with gr.Tab(elem_classes="feedback", label="Editable Blob"):
                        editable_blob = gr.Image(label="Editable Blob", height=320, interactive=False, container=True)
                with gr.Column():
                    with gr.Tab(elem_classes="feedback", label="Edited Preview"):
                        edited_result_gallery = gr.Gallery(label='Edited Preview', height=320, elem_id="gallery", show_label=True, interactive=False, preview=True)
            

            gr.Markdown("### Click to adjust the target blob size 👇", show_label=False)
            with gr.Row():
                with gr.Group():
                    resize_blob_slider_maintain_aspect_ratio = gr.Slider(label="Resize Blob (Maintain Aspect Ratio)", minimum=0.1, maximum=2, step=0.05, value=1)


            with gr.Row():
                undo_blob_button = gr.Button('🔙 Undo Blob', elem_id="undo_btnBlob", scale=1)
                reset_blob_button = gr.Button('🔄 Reset Blob', elem_id="reset_btnBlob", scale=1)

            gr.Markdown("### Click to adjust the initial blob size to ensure it surrounds the object👇", show_label=False)
            with gr.Group():
                with gr.Row():
                    resize_init_blob_slider = gr.Slider(label="Resize Initial Blob (Maintain Aspect Ratio)", minimum=0.1, maximum=2, step=0.05, value=1, scale=4)
                with gr.Row():
                    remove_blob_box = gr.Checkbox(label="Remove Blob", value=False, scale=1)


            gr.Markdown("### Click to achieve more edit types, such as single-sided resize, composition, etc. 👇", show_label=False)
            with gr.Accordion("More Edit Types", open=False, elem_id="accordion3"):

                gr.Markdown("### slide to achieve single-sided resize and rotation", show_label=False)
                with gr.Group():
                    with gr.Row():
                        resize_blob_slider_along_long_axis = gr.Slider(label="Resize Blob (Along Long Axis)", minimum=0, maximum=2, step=0.05, value=1)
                        resize_blob_slider_along_short_axis = gr.Slider(label="Resize Blob (Along Short Axis)", minimum=0, maximum=2, step=0.05, value=1)
                with gr.Row():
                    rotation_blob_slider = gr.Slider(label="Rotate Blob (Clockwise)", minimum=-180, maximum=180, step=1, value=0)

                gr.Markdown("### slide to adjust the initial blob (single-sided)", show_label=False)
                with gr.Group():
                    with gr.Row():
                        resize_init_blob_slider_long_axis = gr.Slider(label="Resize Initial Blob (Long Axis)", minimum=0, maximum=2, step=0.01, value=1)
                        resize_init_blob_slider_short_axis = gr.Slider(label="Resize Initial Blob (Short Axis)", minimum=0, maximum=2, step=0.01, value=1)

                gr.Markdown("### 🎨 Click to set the initial blob and upload object image for compositional generation👇", show_label=False)
                with gr.Accordion("Compositional Generation", open=False, elem_id="accordion5"):
                    with gr.Row():
                        init_ellipse_parameter = gr.Textbox(label="Initial Ellipse", value="[0.5, 0.5, 0.2, 0.2, 180]", scale=4)
                        init_ellipse_button = gr.Button("Set Initial Ellipse", elem_id="set_init_ellipse_btn", scale=1)

                    with gr.Row(elem_id="Image"):
                        with gr.Tab(elem_classes="feedback1", label="User-specified Object Image"):
                            init_object_image = gr.Image(type="numpy", label="User-specified Object Image", height=320)
                
    
            gr.Markdown("## **Step 4: 🚀 Run Generation**", show_label=False)

            run_button = gr.Button("🚀 Run Generation",elem_id="btn")

            with gr.Row():
                with gr.Tab(elem_classes="feedback", label="Results"):
                    results_gallery = gr.Gallery(label='Results', height=320, elem_id="gallery", show_label=True, interactive=False, preview=True)

    eg_index = gr.Textbox(label="Example Index", value="", visible=False)
    with gr.Row():
        examples_inputs = [
                        input_image, 
                        scene_prompt, 
                        blobnet_control_strength, 
                        blobnet_control_guidance_start, 
                        blobnet_control_guidance_end, 
                        seed,
                        eg_index,
                        ]
        examples_outputs = [
            object_image_gallery, 
            ori_result_gallery, 
            editable_blob, 
            edited_result_gallery,
            results_gallery,
            ellipse_lists,
            tracking_points,
            original_image,
            remove_blob_box,
        ]
        def process_example(input_image, 
                    scene_prompt, 
                    blobnet_control_strength, 
                    blobnet_control_guidance_start, 
                    blobnet_control_guidance_end, 
                    seed,
                    eg_index):
        
            eg_index = int(eg_index)
            
            # Force reload images from disk each time
            object_image_gallery = [Image.open(path).copy() for path in OBJECT_IMAGE_GALLERY[eg_index]]
            ori_result_gallery = [Image.open(path).copy() for path in ORI_RESULT_GALLERY[eg_index]]
            editable_blob = Image.open(EDITABLE_BLOB[eg_index]).copy()
            edited_result_gallery = [Image.open(path).copy() for path in EDITED_RESULT_GALLERY[eg_index]]
            results_gallery = [path for path in RESULTS_GALLERY[eg_index]]  # Paths only
            
            # Deep copy mutable data structures
            ellipse_lists = copy.deepcopy(ELLIPSE_LISTS[eg_index])
            tracking_points = copy.deepcopy(TRACKING_POINTS[eg_index])
            
            # Force reload input image
            original_image = np.array(Image.open(INPUT_IMAGE[eg_index]).copy())
            remove_blob_box = REMOVE_STATE[eg_index]

            return object_image_gallery, ori_result_gallery, editable_blob, edited_result_gallery, results_gallery, ellipse_lists, tracking_points, original_image, remove_blob_box

        example = gr.Examples(
            label="Quick Example", 
            examples=EXAMPLES,
            inputs=examples_inputs,
            outputs=examples_outputs,
            fn=process_example,
            examples_per_page=10,
            cache_examples=False,
            run_on_click=True,
            
        )

    with gr.Row():
        gr.Markdown(citation)


    ## initial
    initial_output = [
                       input_image, 
                       original_image, 
                       editable_blob,
                       selected_points, 
                       tracking_points, 
                       ellipse_lists, 
                       ori_result_gallery,
                       object_image_gallery,
                       edited_result_gallery, 
                       results_gallery,
                       blobnet_control_strength,
                       blobnet_control_guidance_start,
                       blobnet_control_guidance_end,
                       resize_blob_slider_maintain_aspect_ratio,
                       resize_blob_slider_along_long_axis,
                       resize_blob_slider_along_short_axis,
                       rotation_blob_slider,
                       resize_init_blob_slider,
                       resize_init_blob_slider_long_axis,
                       resize_init_blob_slider_short_axis,
                       init_ellipse_parameter,
                       init_object_image,
                       remove_blob_box,
                       ]

    input_image.upload(
        initialize_img,
        [input_image],
        initial_output
    )
    
    ## select point
    input_image.select(
        get_point,
        [original_image, selected_points],
        [input_image, original_mask],
    )
    
    undo_seg_button.click(
        undo_seg_points,
        [original_image, selected_points],
        [input_image, original_mask]
    )

    ## blob image and tracking points: move
    editable_blob.select(
        add_tracking_points,
        [original_image, tracking_points, ellipse_lists],
        [editable_blob, tracking_points, ellipse_lists, edited_result_gallery]
    )


    ## undo, reset and save blob
    undo_blob_button.click(
        undo_blob_points,
        [original_image, tracking_points, ellipse_lists],
        [editable_blob, tracking_points, ellipse_lists, edited_result_gallery, resize_blob_slider_maintain_aspect_ratio, resize_blob_slider_along_long_axis, resize_blob_slider_along_short_axis, rotation_blob_slider]
    )

    reset_blob_button.click(
        reset_blob_points,
        [original_image, tracking_points, ellipse_lists],
        [editable_blob, tracking_points, ellipse_lists, edited_result_gallery, resize_blob_slider_maintain_aspect_ratio]
    )


    ## generate blob
    generate_blob_button.click(fn=generate_blob, 
                                inputs=[original_image, original_mask, selected_points, ellipse_lists], 
                                outputs=[editable_blob, ori_result_gallery, object_image_gallery, ellipse_lists, tracking_points, edited_result_gallery, resize_blob_slider_maintain_aspect_ratio, resize_blob_slider_along_long_axis, resize_blob_slider_along_short_axis, rotation_blob_slider, resize_init_blob_slider, resize_init_blob_slider_long_axis, resize_init_blob_slider_short_axis, init_ellipse_parameter, init_object_image])



    ## resize blob
    resize_blob_slider_maintain_aspect_ratio.release(
        resize_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, resize_blob_slider_maintain_aspect_ratio, resize_blob_maintain_aspect_ratio_state, edited_result_gallery, remove_blob_box],
        [editable_blob, ellipse_lists, edited_result_gallery, resize_blob_slider_maintain_aspect_ratio]
    )

    resize_blob_slider_along_long_axis.release(
        resize_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, resize_blob_slider_along_long_axis, resize_blob_along_long_axis_state, edited_result_gallery, remove_blob_box],
        [editable_blob, ellipse_lists, edited_result_gallery, resize_blob_slider_along_long_axis]
    )

    resize_blob_slider_along_short_axis.release(
        resize_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, resize_blob_slider_along_short_axis, resize_blob_along_short_axis_state, edited_result_gallery, remove_blob_box],
        [editable_blob, ellipse_lists, edited_result_gallery, resize_blob_slider_along_short_axis]
    )

    ## rotate blob
    rotation_blob_slider.release(
        rotate_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, rotation_blob_slider],
        [editable_blob, ellipse_lists, edited_result_gallery, rotation_blob_slider]
    )


    remove_blob_box.change(
        remove_blob_box_func,
        [editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, remove_blob_box],
        [editable_blob, ellipse_lists, edited_result_gallery, ori_result_gallery, resize_blob_slider_maintain_aspect_ratio]
    )

    ## resize init blob
    resize_init_blob_slider.release(
        resize_start_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, resize_init_blob_slider, resize_blob_maintain_aspect_ratio_state],
        [editable_blob, ellipse_lists, edited_result_gallery, ori_result_gallery, resize_init_blob_slider]
    )

    resize_init_blob_slider_long_axis.release(
        resize_start_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, resize_init_blob_slider_long_axis, resize_blob_along_long_axis_state],
        [editable_blob, ellipse_lists, edited_result_gallery, ori_result_gallery, resize_init_blob_slider_long_axis]
    )

    resize_init_blob_slider_short_axis.release(
        resize_start_blob,
        [editable_blob, original_image, tracking_points, ellipse_lists, ori_result_gallery, resize_init_blob_slider_short_axis, resize_blob_along_short_axis_state],
        [editable_blob, ellipse_lists, edited_result_gallery, ori_result_gallery, resize_init_blob_slider_short_axis]
    )

    ## set initial ellipse
    init_ellipse_button.click(
        set_init_ellipse,
        inputs=[original_image, original_mask, edited_result_gallery, ellipse_lists, tracking_points, editable_blob, ori_result_gallery, init_ellipse_parameter], 
        outputs=[editable_blob, edited_result_gallery, ellipse_lists, tracking_points, ori_result_gallery, init_ellipse_parameter]
    )

    ## upload user-specified object image
    init_object_image.upload(
        upload_object_image,
        [init_object_image, edited_result_gallery, remove_blob_box],
        [object_image_gallery, remove_blob_box]
    )

    ## run BlobEdit
    ips = [
          original_image,
          scene_prompt, 
          ori_result_gallery, 
          object_image_gallery,
          edited_result_gallery, 
          ellipse_lists, 
          blobnet_control_strength, 
          blobnet_control_guidance_start,
          blobnet_control_guidance_end,
          remove_blob_box,
          num_samples, 
          seed,
          guidance_scale,
          num_inference_steps,
          # for save
          editable_blob,
          resize_blob_slider_maintain_aspect_ratio,
          resize_blob_slider_along_long_axis,
          resize_blob_slider_along_short_axis,
          rotation_blob_slider,
          resize_init_blob_slider,
          resize_init_blob_slider_long_axis,
          resize_init_blob_slider_short_axis,
          tracking_points,
          ]
    run_button.click(
        run_function,
        ips,
        [results_gallery]
    )


## if have a localhost access error, try to use the following code
demo.launch(server_name="0.0.0.0", server_port=12346)
# demo.launch()