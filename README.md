# SAM2 Bookshelf Segmentation (Simple)

Scripts for running **Segment Anything 2 (SAM2)** on your own images to segment books/objects and export colored instance overlays (each object gets a unique color). Works with SAM2/SAM2.1 configs and supports both automatic and point-prompt modes for quick experiments or more control. Includes a simple CLI for batch processing folders, saving masks/overlays/metadata, and using your own fine-tuned checkpoints.

## 1) Setup (Conda, Python 3.10)

```bash
# create env
conda create -n sam2ocr python=3.10 -y
conda activate sam2ocr

# install PyTorch (pick one that matches your CUDA)
# example for CUDA 12:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# install SAM2 (from your local clone)
pip install -e sam2/  # path to the cloned facebookresearch/sam2 repo

# other deps
pip install opencv-python-headless matplotlib tqdm numpy hydra-core
```

> Tip: verify GPU with `python -c "import torch; print(torch.cuda.is_available())"`.

## 2) Files You Need

* **Checkpoint**: e.g. `sam2_hiera_tiny.pt` (or your fine-tuned `.pt` in `sam2_logs/.../checkpoints/last.ckpt`)
* **Config**: e.g. `configs/sam2.1/sam2.1_hiera_tiny.yaml` (or the one that matches your checkpoint)
* **Images**: a folder of `.jpg/.png` you want to segment

## 3) Quick Inference (Auto masks, colored overlay)

```bash
python infer_sam2_local_auto.py \
  --images "/path/to/images/*.jpg" \
  --out "/path/to/out" \
  --model_cfg "configs/sam2.1/sam2.1_hiera_tiny.yaml" \
  --ckpt "/path/to/checkpoint.pt" \
  --device "cuda" \
  --save-overlay
```

Outputs:

* `out/masks/*.png` — merged binary masks
* `out/overlays/*_overlay.png` — original image with **rainbow** instance colors
* `out/meta/*.json` — small metadata (counts, etc.)

## 4) (Optional) Point-Prompt Mode

```bash
python infer_sam2_local_auto.py \
  --images "/path/to/images/*.jpg" \
  --out "/path/to/out_points" \
  --model_cfg "configs/sam2.1/sam2.1_hiera_tiny.yaml" \
  --ckpt "/path/to/checkpoint.pt" \
  --device "cuda" \
  --mode "points" \
  --points "center" \
  --save-overlay
```

## 5) Repo Layout (minimal)

```
.
├── infer_sam2_local_auto.py   # simple CLI for auto/points with colored overlays
├── README.md
└── (your) sam2/               # installed SAM2 source (facebookresearch/sam2)
```

## 6) License

This repo’s code: MIT.
SAM2 is by Meta; follow their license in `sam2/`.

CUDA errors: match PyTorch + CUDA build; free VRAM (close browsers, viewers).

Import error (shadowing): run scripts outside the sam2/ parent folder after pip install -e sam2/.
