import os
from huggingface_hub import snapshot_download

# download blobctrl models
BlobCtrl_path = "examples/blobctrl/models"
if not (os.path.exists(f"{BlobCtrl_path}/blobnet") and os.path.exists(f"{BlobCtrl_path}/unet_lora")):
    BlobCtrl_path = snapshot_download(
        repo_id="Yw22/BlobCtrl",
        local_dir=BlobCtrl_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"BlobCtrl checkpoints downloaded to {BlobCtrl_path}")

# download stable-diffusion-v1-5
StableDiffusion_path = "examples/blobctrl/models/stable-diffusion-v1-5"
if not os.path.exists(StableDiffusion_path):
    StableDiffusion_path = snapshot_download(
        repo_id="sd-legacy/stable-diffusion-v1-5",
        local_dir=StableDiffusion_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"StableDiffusion checkpoints downloaded to {StableDiffusion_path}")

# download dinov2-large
Dino_path = "examples/blobctrl/models/dinov2-large"
if not os.path.exists(Dino_path):
    Dino_path = snapshot_download(
        repo_id="facebook/dinov2-large",
        local_dir=Dino_path,
        token=os.getenv("HF_TOKEN"),
    )
print(f"Dino checkpoints downloaded to {Dino_path}")