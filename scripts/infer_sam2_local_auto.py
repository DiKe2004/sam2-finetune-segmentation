#!/usr/bin/env python3
import os, glob, argparse, json, pathlib
import numpy as np
import cv2
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ---------------------------- utils ----------------------------

def bgr_to_rgb_contig(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(path: str):
    p = pathlib.Path(path)
    if p.is_dir():
        pats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for pt in pats: files += glob.glob(str(p / pt))
        return sorted(files)
    else:
        return sorted(glob.glob(path))

def unique_stem(path):
    return pathlib.Path(path).stem

def as_contig_u8(arr_bool_like) -> np.ndarray:
    """bool/0-1 → uint8 {0,255} and contiguous"""
    m = (np.asarray(arr_bool_like) > 0).astype(np.uint8) * 255
    return np.ascontiguousarray(m)

def distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """
    Return n distinct BGR colors (uint8). Spread hues in HSV.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)  # for slight value jitter
    cols = []
    for k in range(n):
        h = int(179 * k / max(1, n))                # 0..179
        s = 200 + rng.integers(0, 55)               # 200..254
        v = 200 + rng.integers(0, 55)               # 200..254
        hsv = np.uint8([[[h, s, v]]])               # (1,1,3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(bgr)
    return np.stack(cols, axis=0)

def overlay_instances(image_bgr: np.ndarray,
                      masks_u8: list[np.ndarray],
                      alpha: float = 0.6,
                      draw_edges: bool = True,
                      seed: int = 42) -> np.ndarray:
    """
    Blend per-instance colored masks over the image.
    masks_u8: list of uint8 masks 0/255, HxW
    """
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    n = len(masks_u8)
    if n == 0:
        return overlay

    colors = distinct_colors(n, seed=seed)  # BGR

    # paint each mask
    for i, m in enumerate(masks_u8):
        if m is None: continue
        m = (m > 0).astype(np.uint8)  # 0/1
        if m.sum() == 0: continue

        col = colors[i].astype(np.uint8)
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        color_img[..., 0] = col[0]
        color_img[..., 1] = col[1]
        color_img[..., 2] = col[2]

        # alpha blend on mask region
        mask3 = np.dstack([m]*3)
        overlay = np.where(mask3, (overlay*(1-alpha) + color_img*alpha).astype(np.uint8), overlay)

        # optional crisp edge
        if draw_edges:
            edges = cv2.Canny(m*255, 0, 1)  # thin outline
            overlay[edges > 0] = (0, 0, 0)  # black edge; change as you want

    return overlay


# ----------------------- inference helpers ----------------------

@torch.no_grad()
def infer_auto(model, image_bgr,
               pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=0):
    rgb = bgr_to_rgb_contig(image_bgr)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return generator.generate(rgb)  # list of dicts with "segmentation"

@torch.no_grad()
def infer_point_prompt(predictor: SAM2ImagePredictor, image_bgr, points="center"):
    rgb = bgr_to_rgb_contig(image_bgr)
    predictor.set_image(rgb)

    H, W = rgb.shape[:2]
    if points == "center":
        pts = np.array([[[W // 2, H // 2]]], dtype=np.int64)
    elif points == "grid4":
        pts = np.array(
            [
                [[W // 3,     H // 3]],
                [[2 * W // 3, H // 3]],
                [[W // 3,     2 * H // 3]],
                [[2 * W // 3, 2 * H // 3]],
            ],
            dtype=np.int64,
        )
    else:
        raise ValueError("points must be 'center' or 'grid4'")

    labels = np.ones((pts.shape[0], 1), dtype=np.int64)

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(pts, labels), boxes=None, masks=None
    )

    high_res_features = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
    image_embed = predictor._features["image_embed"][-1].unsqueeze(0)

    low_res_masks, iou_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=(pts.shape[0] > 1),
        high_res_features=high_res_features,
    )

    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
    prd_masks = torch.sigmoid(prd_masks)
    best_idx = torch.argmax(iou_scores[:, 0]).item()
    best = (prd_masks[best_idx, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
    return best


# ------------------------------ main ----------------------------

def main():
    ap = argparse.ArgumentParser("SAM2 Local Inference (Rainbow Overlay)")
    ap.add_argument("--images", required=True, help="Folder or glob to images")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--model_cfg", required=True,
                    help="Hydra config (e.g. 'configs/sam2.1/sam2.1_hiera_base_plus.yaml' or absolute path)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt/.pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mode", default="auto", choices=["auto", "points"])
    ap.add_argument("--points", default="center", choices=["center", "grid4"])
    ap.add_argument("--save-overlay", action="store_true", help="Save colored overlays (recommended)")
    ap.add_argument("--auto-iou", type=float, default=0.86)
    ap.add_argument("--auto-stability", type=float, default=0.92)
    ap.add_argument("--auto-min-area", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out)
    out_mask_dir = os.path.join(args.out, "masks")
    out_ovl_dir  = os.path.join(args.out, "overlays")
    out_meta_dir = os.path.join(args.out, "meta")
    ensure_dir(out_mask_dir); ensure_dir(out_meta_dir)
    if args.save_overlay: ensure_dir(out_ovl_dir)

    # Build model once
    sam2_model = build_sam2(args.model_cfg, args.ckpt, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model) if args.mode == "points" else None

    files = list_images(args.images)
    if not files:
        raise RuntimeError(f"No images found for {args.images}")
    print(f"[INFO] Found {len(files)} images.")

    for idx, f in enumerate(files, 1):
        img_bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read: {f}")
            continue

        stem = unique_stem(f)

        if args.mode == "auto":
            results = infer_auto(
                sam2_model, img_bgr,
                pred_iou_thresh=args.auto_iou,
                stability_score_thresh=args.auto_stability,
                min_mask_region_area=args.auto_min_area,
            )

            # list of instance masks (uint8 0/255), keep separate
            inst_masks = []
            for r in results:
                seg = r.get("segmentation", None)
                if seg is None: continue
                inst_masks.append(as_contig_u8(seg))

            # merged binary mask (for any later pipeline)
            if len(inst_masks) == 0:
                merged = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            else:
                merged = np.maximum.reduce(inst_masks)

            cv2.imwrite(os.path.join(out_mask_dir, f"{stem}.png"), merged)

            # colorful overlay (rainbow)
            if args.save_overlay:
                ovl = overlay_instances(img_bgr, inst_masks, alpha=0.6, draw_edges=True)
                cv2.imwrite(os.path.join(out_ovl_dir, f"{stem}_overlay.png"), ovl)

            meta = {"image": f, "num_masks": len(inst_masks)}
            with open(os.path.join(out_meta_dir, f"{stem}.json"), "w") as w:
                json.dump(meta, w)

        else:
            # single mask from prompt → still color it so it stands out
            mask = infer_point_prompt(predictor, img_bgr, points=args.points)
            cv2.imwrite(os.path.join(out_mask_dir, f"{stem}.png"), mask)
            if args.save_overlay:
                ovl = overlay_instances(img_bgr, [mask], alpha=0.6, draw_edges=True)
                cv2.imwrite(os.path.join(out_ovl_dir, f"{stem}_overlay.png"), ovl)

        if idx % 10 == 0:
            print(f"[INFO] {idx}/{len(files)} processed")

    print("[DONE] Inference complete. Outputs in:", args.out)


if __name__ == "__main__":
    main()
