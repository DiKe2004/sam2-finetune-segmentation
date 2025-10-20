# SAM2 Bookshelf Segmentation (Simple)

Lightweight repo to run Segment Anything 2 (SAM2) for bookshelf/object segmentation on your own images.
Includes colored instance overlays so you can see each segmented object clearly.

1) Setup (Conda, Python 3.10)
#create env
conda create -n sam2ocr python=3.10 -y
conda activate sam2ocr

#install PyTorch (pick one that matches your CUDA)
#example for CUDA 12:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

#install SAM2 (from your local clone)
pip install -e sam2/  # path to the cloned facebookresearch/sam2 repo

#other deps
pip install opencv-python-headless matplotlib tqdm numpy hydra-core


Tip: verify GPU with python -c "import torch; print(torch.cuda.is_available())".

2) Files You Need

Checkpoint: e.g. sam2_hiera_tiny.pt (or your fine-tuned .pt in sam2_logs/.../checkpoints/last.ckpt)

Config: e.g. configs/sam2.1/sam2.1_hiera_tiny.yaml (or the one that matches your checkpoint)

Images: a folder of .jpg/.png you want to segment

Quick Inference (Auto masks, colored overlay)
python infer_sam2_local_auto.py \
  --images "/path/to/images/*.jpg" \
  --out "/path/to/out" \
  --model_cfg "configs/sam2.1/sam2.1_hiera_tiny.yaml" \
  --ckpt "/path/to/checkpoint.pt" \
  --device "cuda" \
  --save-overlay


Outputs:

out/masks/*.png — merged binary masks

out/overlays/*_overlay.png — original image with rainbow instance colors

out/meta/*.json — small metadata (counts, etc.)
