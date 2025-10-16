# Bookshelf SAM2 — Fine-Tuning & Inference

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.x-red.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Minimal, reproducible pipeline to **fine-tune Segment Anything 2 (SAM2)** on bookshelf images and run **local inference** with colorful instance overlays.

<p align="center">
  <img src="docs/sample_overlay.png" width="70%" />
</p>

## 🧭 Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Checkpoints & Configs](#checkpoints--configs)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Troubleshooting / FAQ](#troubleshooting--faq)
- [Cite & License](#cite--license)

---

## Features
- SAM2 **fine-tuning** (tiny/small configs for 8–12 GB GPUs).
- **Local** inference (auto-masks or point prompts).
- **Rainbow overlays** to visualize distinct instances.
- Hydra configs, conda env, and reproducible commands.

---

## Quick Start

```bash
# Create env (Python 3.10)
conda env create -f environment.yml
conda activate sam2ocr

# Install SAM2 (editable)
cd sam2 && pip install -e . && cd ..

# Run inference (auto masks + color overlays)
python scripts/infer_sam2_local_auto.py \
  --images "data/bookshelf/valid/images/*.jpg" \
  --out runs/infer_demo \
  --model_cfg sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --ckpt checkpoints/finetuned_bookshelf.pt \
  --device cuda \
  --save-overlay
