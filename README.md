Here you go — a single, copy-pasteable `README.md` that includes everything (overview, setup, training/inference, troubleshooting, plus embedded Contributing, Code of Conduct, and Security sections). You can tweak names/paths/emails and push it as-is.

````markdown
# Bookshelf-SAM2: Lightweight SAM 2 Segmentation for Bookshelves

Fast, GPU-friendly segmentation for bookshelf images using **Segment Anything 2 (SAM2)**.  
This repo includes:
- Local inference scripts (auto + point prompts)
- Rainbow instance overlays to visualize per-book segments
- Minimal fine-tuning workflow on your own masks
- Reproducible environment hints (CUDA/PyTorch)

> Works on RTX A5000 (24GB) with PyTorch CUDA builds.

---

## ✨ Features

- **Auto mask generation** or **point-prompted** segmentation
- **Rainbow overlays**: unique color per segment to see individual books
- **VRAM-aware** configs (tiny/small backbones, AMP, grad accumulation)
- **Plain Python scripts** (no heavy frameworks required)

---

## 📦 Quick Start

### 1) Create/activate environment
```bash
# Example with conda (Python 3.10 recommended)
conda create -n sam2ocr python=3.10 -y
conda activate sam2ocr

# Install PyTorch matching your CUDA (example: CUDA 12.x build)
# See https://pytorch.org/get-started/locally/ for the exact command.
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
````

### 2) Install SAM2 (as a package)

```bash
# In your project root (NOT the parent folder of the repo named 'sam2')
pip install -e sam2
```

### 3) Get SAM2 checkpoints

Place your checkpoint (e.g. `sam2.1_hiera_tiny.pt`) under:

```
<repo-root>/checkpoints/
```

### 4) Run local inference (auto mode)

```bash
python infer_sam2_local_auto.py \
  --images "/path/to/images/*.jpg" \
  --out "/path/to/out_dir" \
  --model_cfg "configs/sam2.1/sam2.1_hiera_tiny.yaml" \
  --ckpt "checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --save-overlay
```

* Outputs:

  * `out_dir/masks/*.png` (binary or merged masks)
  * `out_dir/overlays/*_overlay.png` (rainbow instance overlay if enabled)
  * `out_dir/meta/*.json` (counts, etc.)

> If you get a **Hydra MissingConfigException**, the path in `--model_cfg` is wrong. Use a valid file from the repo’s `configs/sam2.1/` directory.

---

## 🧠 Point-Prompted Inference

```bash
python infer_sam2_local_auto.py \
  --images "/path/to/images/*.jpg" \
  --out "/path/to/out_dir" \
  --model_cfg "configs/sam2.1/sam2.1_hiera_tiny.yaml" \
  --ckpt "checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --mode points \
  --points grid4 \
  --save-overlay
```

* `--points center` places a single positive point in the image center.
* `--points grid4` places four positive points (helps cover multiple books).

---

## 🏋️ Fine-Tuning (Optional)

You’ll need **paired images and masks**:

```
/data/bookshelf_sam2_pairs/
  train/
    images/*.jpg|png
    masks/*.png         # single-object or merged object masks (white=object)
  valid/
    images/*.jpg|png
    masks/*.png
```

High-level steps (script not shown here to keep README short):

1. Load `sam2.1_hiera_tiny.yaml` + checkpoint.
2. Freeze image encoder; train prompt encoder + mask decoder (lower VRAM).
3. Use AMP + gradient accumulation.
4. Save checkpoints to `weights/` periodically.

> Fine-tuning improves performance on your domain (your shelf, lighting, camera).

---

## 🧪 Verifying CUDA + PyTorch

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

* If `False`, install a CUDA-enabled PyTorch matching your driver/CUDA.
* If you see `expandable_segments=True` errors, unset the env var:

  ```bash
  unset PYTORCH_CUDA_ALLOC_CONF
  ```

---

## 🛠️ Troubleshooting

**Hydra Missing Config**

* Ensure `--model_cfg` is a valid YAML in `configs/sam2.1/` (e.g. `sam2.1_hiera_tiny.yaml`).
* Don’t pass absolute paths missing a leading slash (e.g. `home/user/...` → should be `/home/user/...`).

**“You’re likely running Python from the parent directory of the sam2 repo”**

* Don’t run scripts from the parent of the folder named `sam2`.
* `pip install -e sam2` then run Python from the *project root* or home dir.

**Illegal instruction / kernel errors**

* Match your PyTorch build to your CUDA/driver.
* Try `--device cpu` to isolate CUDA issues.

**Torch OOM**

* Use `sam2.1_hiera_tiny.yaml` and AMP; reduce image size; use fewer points; close other GPU apps.

**Negative strides error**

* Ensure images are contiguous; in code, convert masks with `np.ascontiguousarray`.

---

## 📁 Repo Layout (suggested)

```
.
├── checkpoints/                 # put your .pt here
├── configs/
│   └── sam2.1/
│       ├── sam2.1_hiera_tiny.yaml
│       └── ...
├── sam2/                        # SAM2 package (installed in editable mode)
├── scripts/                     # (optional) data prep / finetune helpers
├── infer_sam2_local_auto.py     # main inference script (auto/points + rainbow)
├── LICENSE
└── README.md
```

---

## 📜 License

MIT (or your choice). See `LICENSE`.

---

## 🤝 Contributing

We welcome issues and PRs!

1. Search existing issues.
2. Open a new issue to discuss significant changes.
3. Follow the coding style and PR checklist below.

**Dev setup**

```bash
conda activate sam2ocr
pip install -r requirements-dev.txt  # optional (ruff, pytest)
```

**Style & tests**

```bash
ruff check .
pytest -q
```

**PR Checklist**

* [ ] Clear title/description
* [ ] Docs/README updated
* [ ] Tests or manual validation steps
* [ ] Lint/test pass

*By contributing, you agree your contributions are licensed under the project license.*

---

## 🧭 Code of Conduct

We strive for a welcoming, inclusive community.

* Be respectful and constructive.
* No harassment or discrimination.
* Keep feedback about the work, not the person.

**Reporting**
Email **<[maintainer-email@example.com](mailto:maintainer-email@example.com)>** for any conduct concerns.
We’ll review promptly and handle confidentially.

(Adapted from the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1.)

---

## 🔐 Security Policy

If you find a vulnerability:

* **Do not** open a public issue.
* Email **<[security-contact@example.com](mailto:security-contact@example.com)>** with:

  * Description + steps to reproduce/PoC
  * Affected versions/commits
  * Environment details

We’ll acknowledge within **3 business days**, investigate, patch, and—if you agree—credit you in release notes.

---

## 🙌 Acknowledgements

* **Meta FAIR** for [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2).
* Open-source community & PyTorch.

```

**Tip:** after you paste this, rename the repo, fix any paths (like `configs/sam2.1/...` and your `checkpoints/` filename), and replace the placeholder emails.
::contentReference[oaicite:0]{index=0}
```
