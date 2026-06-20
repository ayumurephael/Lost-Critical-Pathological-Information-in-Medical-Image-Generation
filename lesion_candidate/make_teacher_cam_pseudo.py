from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .data import read_manifest
from .models import MUISStyleEncoder
from .preprocess import bbox_to_original, load_gray, mask_to_bbox, mask_to_original, resize_image, robust_normalize, retina_support, save_mask, save_overlay
from .pseudo import PseudoConfig, edge_touch_fraction, postprocess_score, quality_score


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(k.startswith("_orig_mod.") for k in state):
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


def load_teacher(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu")
    dim_zs = int(ckpt.get("dim_zs", 64))
    dim_zc = int(ckpt.get("dim_zc", 8))
    base = int(ckpt.get("base", 64))
    image_size = int(ckpt.get("image_size", 192))
    model = MUISStyleEncoder(dim_zs=dim_zs, dim_zc=dim_zc, base=base)
    model.load_state_dict(_strip_compile_prefix(ckpt["encoder"]), strict=True)
    model.eval()
    return model, image_size, dim_zc


def cam_to_score(cam: torch.Tensor, zc_logits: torch.Tensor, mode: str = "pred") -> np.ndarray:
    cam = torch.relu(cam)
    if mode == "max":
        score = cam.max(dim=1, keepdim=True).values
    elif mode == "mean":
        score = cam.mean(dim=1, keepdim=True)
    else:
        ids = torch.argmax(zc_logits, dim=1)
        score = cam[torch.arange(cam.size(0), device=cam.device), ids][:, None]
    score = score[0, 0].detach().float().cpu().numpy()
    lo, hi = np.percentile(score, [2, 99.5])
    if hi <= lo + 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def generate_teacher_cam_pseudo(
    checkpoint: Path,
    manifest: Path,
    out_dir: Path,
    cfg: PseudoConfig,
    cam_mode: str = "pred",
    max_pairs: int | None = None,
    overlays: int = 200,
) -> Path:
    rows = read_manifest(manifest)
    if max_pairs is not None:
        rows = rows[:max_pairs]
    model, teacher_size, dim_zc = load_teacher(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    masks_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, object]] = []
    target_size = (cfg.target_width, cfg.target_height)

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc="teacher-cam pseudo")):
            post_original = robust_normalize(load_gray(Path(row["post_path"])))
            post_target, target_meta = resize_image(post_original, target_size)
            post_teacher, _ = resize_image(post_original, (teacher_size, teacher_size))
            x = torch.from_numpy(post_teacher[None, None].astype(np.float32)).to(device)
            _, zc_logits, _, cam, _ = model(x, return_cam=True)
            if cam is None:
                raise RuntimeError("Teacher did not return CAM tensors.")
            score_teacher = cam_to_score(cam, zc_logits, mode=cam_mode)
            score = cv2.resize(score_teacher, target_size, interpolation=cv2.INTER_LINEAR)
            support = retina_support(post_target)
            score = score * support.astype(np.float32)
            mask, info = postprocess_score(score, support, cfg)
            quality = quality_score(mask, score, support)
            if quality < cfg.min_quality:
                mask = np.zeros_like(mask, dtype=bool)
                info["empty_reason"] = info.get("empty_reason") or "below_min_quality"

            bbox_small = mask_to_bbox(mask)
            mask_original = mask_to_original(mask, target_meta)
            bbox_original = bbox_to_original(bbox_small, target_meta)
            mask_path = masks_dir / f"{row['pair_id']}.png"
            save_mask(mask_original, mask_path)
            if idx < overlays:
                save_overlay(post_original, mask_original, bbox_original, overlay_dir / f"{row['pair_id']}.png")
            num = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1 if mask.any() else 0
            out_rows.append(
                {
                    **row,
                    "pseudo_mask_path": str(mask_path),
                    "bbox_x0": bbox_original[0] if bbox_original else "",
                    "bbox_y0": bbox_original[1] if bbox_original else "",
                    "bbox_x1": bbox_original[2] if bbox_original else "",
                    "bbox_y1": bbox_original[3] if bbox_original else "",
                    "pseudo_quality": float(quality),
                    "pseudo_area_fraction": float(mask.mean()),
                    "pseudo_components": int(num),
                    "pseudo_edge_touch_fraction": float(edge_touch_fraction(mask)),
                    "pseudo_empty_reason": str(info.get("empty_reason", "")),
                    "pseudo_source": "teacher_cam",
                    "cam_mode": cam_mode,
                    "teacher_dim_zc": dim_zc,
                }
            )

    out_csv = out_dir / "pseudo_manifest.csv"
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
    parser.add_argument("--threshold-percentile", type=float, default=96.5)
    parser.add_argument("--min-area-fraction", type=float, default=0.0010)
    parser.add_argument("--morph-radius", type=int, default=2)
    parser.add_argument("--min-quality", type=float, default=0.14)
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--target-height", type=int, default=496)
    parser.add_argument("--cam-mode", choices=["pred", "max", "mean"], default="pred")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--overlays", type=int, default=200)
    args = parser.parse_args()

    cfg = PseudoConfig(
        threshold_percentile=args.threshold_percentile,
        min_area_fraction=args.min_area_fraction,
        morph_radius=args.morph_radius,
        min_quality=args.min_quality,
        target_width=args.target_width,
        target_height=args.target_height,
    )
    out = generate_teacher_cam_pseudo(
        args.checkpoint,
        args.manifest,
        args.out_dir,
        cfg,
        cam_mode=args.cam_mode,
        max_pairs=args.max_pairs,
        overlays=args.overlays,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
