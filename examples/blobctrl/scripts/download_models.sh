echo "Downloading BlobCtrl models..."
python examples/blobctrl/download_models.py

echo "Downloading SAM model..."
if [ ! -f "examples/blobctrl/models/sam/sam_vit_h_4b8939.pth" ]; then
    mkdir -p examples/blobctrl/models/sam
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O examples/blobctrl/models/sam/sam_vit_h_4b8939.pth
fi