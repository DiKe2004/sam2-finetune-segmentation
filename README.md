REPO STRUCTURE
.
├─ sam2/                         # SAM-2 library (installed in editable mode)
├─ scripts/
│  ├─ train.yaml                 # your training config (Hydra)
│  ├─ infer_sam2_local_auto.py   # auto-mask inference w/ rainbow overlays
│  └─ infer_sam2_local.py        # (optional) point-prompt inference
├─ checkpoints/                  # sam2_*.pt weights (pretrained & fine-tuned)
├─ data/
│  └─ <your_dataset>/            # train/valid images + masks (PNG)
├─ runs/                         # training outputs (logs, weights)
└─ docs/
   └─ sample_overlay.png         # example visualization (optional)

ENVIROMENT
# 1) Create the env (Python 3.10)
conda create -n sam2.1ft python=3.10 -y
conda activate sam2.1ft

# 2) Install PyTorch (match your CUDA, e.g., 12.1 wheels)
# Visit https://pytorch.org/get-started/locally/ for the command suited to your CUDA.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Project deps
pip install -r requirements.txt  # (make one if you like) or:
pip install opencv-python-headless==4.10.0.84 hydra-core tqdm matplotlib numpy

# 4) Install SAM-2 (editable)
cd sam2
pip install -e .
cd ..

CHECKPOINT
checkpoints/
├─ sam2.1_hiera_tiny.pt
├─ sam2.1_hiera_base_plus.pt
└─ finetuned_bookshelf.pt        # produced by your training

DATA FORMAT
data/your_dataset/
├─ train/
│  ├─ images/*.jpg|png
│  └─ masks/*.png        # single-channel, 0/255
└─ valid/
   ├─ images/*.jpg|png
   └─ masks/*.png

