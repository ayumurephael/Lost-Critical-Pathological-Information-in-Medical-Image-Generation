from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .data import read_manifest
from .models import StudentUNet, main_logits
from .preprocess import load_gray, mask_to_bbox, robust_normalize, resize_image, save_mask, save_overlay


def load_student(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu")
    base = int(ckpt.get("base", 64))
    dropout = float(ckpt.get("dropout", 0.10))
    image_size = tuple(ckpt.get("image_size", (768, 496)))
    deep_supervision = bool(ckpt.get("deep_supervision", True))
    threshold = float(ckpt.get("threshold", 0.5))
    model = StudentUNet(base=base, dropout=dropout, deep_supervision=deep_supervision)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, image_size, threshold


def infer_manifest(
    checkpoint: Path,
    manifest: Path,
    out_dir: Path,
    threshold: float | None = None,
    overlay_limit: int = -1,
) -> Path:
    rows = read_manifest(manifest)
    model, image_size, ckpt_threshold = load_student(checkpoint)
    threshold = ckpt_threshold if threshold is None else threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    masks = out_dir / "masks"
    overlays = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc="infer")):
            pre_original = robust_normalize(load_gray(Path(row["pre_path"])))
            pre, meta = resize_image(pre_original, image_size)
            x = torch.from_numpy(pre[None, None].astype(np.float32)).to(device)
            prob = torch.sigmoid(main_logits(model(x)))[0, 0].cpu().numpy()
            mask_small = prob >= threshold
            mask = cv2.resize(mask_small.astype(np.uint8), (meta.original_width, meta.original_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            bbox = mask_to_bbox(mask)
            mask_path = masks / f"{row['pair_id']}.png"
            save_mask(mask, mask_path)
            if overlay_limit < 0 or idx < overlay_limit:
                save_overlay(pre_original, mask, bbox, overlays / f"{row['pair_id']}.png")
            out_rows.append(
                {
                    **row,
                    "pred_mask_path": str(mask_path),
                    "pred_bbox_x0": bbox[0] if bbox else "",
                    "pred_bbox_y0": bbox[1] if bbox else "",
                    "pred_bbox_x1": bbox[2] if bbox else "",
                    "pred_bbox_y1": bbox[3] if bbox else "",
                    "pred_area_fraction": float(mask.mean()),
                    "pred_prob_mean": float(prob.mean()),
                    "pred_prob_max": float(prob.max()),
                    "threshold": float(threshold),
                }
            )

    out_csv = out_dir / "predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Default: use threshold saved in checkpoint.")
    parser.add_argument("--overlay-limit", type=int, default=-1, help="-1 writes all overlays; 0 writes none.")
    args = parser.parse_args()
    out = infer_manifest(args.checkpoint, args.manifest, args.out_dir, threshold=args.threshold, overlay_limit=args.overlay_limit)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
