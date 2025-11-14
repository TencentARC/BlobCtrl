#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from urllib.request import urlopen, Request
from huggingface_hub import snapshot_download



models_root = Path("models")
models_root.mkdir(parents=True, exist_ok=True)

print(">>> Downloading models for BlobCtrl ...")

# 1) SAM H checkpoint
sam_path = models_root / "sam/sam_vit_h_4b8939.pth"
if sam_path.exists() and sam_path.stat().st_size > 0:
    print("[SAM] already exists, skip.")
else:
    sam_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    print(f"[SAM] downloading {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(sam_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[SAM] saved to {sam_path}")

# 2) BlobCtrl/blobnet 子文件夹
blobnet_dir = models_root / "blobnet"
if blobnet_dir.exists() and any(blobnet_dir.iterdir()):
    print("[Blobnet] already exists, skip.")
else:
    print("[Blobnet] downloading Yw22/BlobCtrl: blobnet/*")
    snapshot_download(
        repo_id="Yw22/BlobCtrl",
        local_dir=str(models_root),
        local_dir_use_symlinks=False,
        allow_patterns=["blobnet/*"],
    )
    print("[Blobnet] done.")

# 3) BlobCtrl/unet_lora 子文件夹
lora_dir = models_root / "unet_lora"
if lora_dir.exists() and any(lora_dir.iterdir()):
    print("[LoRA] already exists, skip.")
else:
    print("[LoRA] downloading Yw22/BlobCtrl: unet_lora/*")
    snapshot_download(
        repo_id="Yw22/BlobCtrl",
        local_dir=str(models_root),
        local_dir_use_symlinks=False,
        allow_patterns=["unet_lora/*"],
    )
    print("[LoRA] done.")

# 4) DINOv2 Large 整仓
dinov2_dir = models_root / "dinov2-large"
if dinov2_dir.exists() and any(dinov2_dir.iterdir()):
    print("[DINOv2-Large] already exists, skip.")
else:
    print("[DINOv2-Large] downloading facebook/dinov2-large")
    snapshot_download(
        repo_id="facebook/dinov2-large",
        local_dir=str(dinov2_dir),
        local_dir_use_symlinks=False,
    )
    print("[DINOv2-Large] done.")

# 5) Stable Diffusion v1.5（白名单+黑名单过滤）
sd15_dir = models_root / "stable-diffusion-v1-5"
if sd15_dir.exists() and any(sd15_dir.iterdir()):
    print("[SD1.5] already exists, skip.")
else:
    print("[SD1.5] downloading runwayml/stable-diffusion-v1-5 (filtered)")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=str(sd15_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "model_index.json",
            "v1-inference.yaml",
            "tokenizer/*",
            "feature_extractor/*",
            "scheduler/*",
            "safety_checker/config.json",
            "safety_checker/model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
        ],
        ignore_patterns=[
            "*pruned*",
            "*emaonly*",
            "*non_ema*",
            "*.ckpt",
            "*.bin",
        ],
    )
    print("[SD1.5] done.")

print("\n✅ All downloads finished! You can now run:")
print("   bash scripts/run_app.sh")
