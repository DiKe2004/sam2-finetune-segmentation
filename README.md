# SAM2 Object & Instance Segmentation

Scripts for running **Segment Anything 2 (SAM2)** on your own images to segment books/objects and export colored instance overlays (each object gets a unique color). Works with SAM2/SAM2.1 configs and supports both automatic and point-prompt modes for quick experiments or more control. Includes a simple CLI for batch processing folders, saving masks/overlays/metadata, and using your own fine-tuned checkpoints.
<img width="1762" height="444" alt="image" src="https://github.com/user-attachments/assets/cddc9369-f8d8-4c52-8279-23d56d0e786a" />


## Installation 

Set up a clean Conda environment named sam2ocr with Python 3.10, then install PyTorch (torch, torchvision, torchaudio) using the official PyTorch wheel index that matches your CUDA version. Next, install SAM2 from your local clone in editable mode (i.e., run pip install -e on the cloned facebookresearch/sam2 folder), and add the common dependencies opencv-python-headless, matplotlib, tqdm, numpy, and hydra-core. Finally, double-check you’re not launching Python from the parent directory of the SAM2 repository to avoid import shadowing.

## Files You Need

* **Checkpoint**: e.g. `sam2_hiera_tiny.pt` (or your fine-tuned `.pt` in `sam2_logs/.../checkpoints/last.ckpt`)
* **Config**: e.g. `configs/sam2.1/sam2.1_hiera_tiny.yaml` (or the one that matches your checkpoint)
* **Images**: a folder of `.jpg/.png` you want to segment

## Quick Inference (Auto masks, colored overlay)

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

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/ed03d05d-1df2-4dfb-aae1-9cd5706d4622" />



## Repo Layout (minimal)

```
.
├── infer_sam2_local_auto.py   # simple CLI for auto/points with colored overlays
├── README.md
└── (your) sam2/               # installed SAM2 source (facebookresearch/sam2)
```

## License

This repo’s code: MIT.
SAM2 is by Meta; follow their license in `sam2/`.

CUDA errors: match PyTorch + CUDA build; free VRAM (close browsers, viewers).

Import error (shadowing): run scripts outside the sam2/ parent folder after pip install -e sam2/.
