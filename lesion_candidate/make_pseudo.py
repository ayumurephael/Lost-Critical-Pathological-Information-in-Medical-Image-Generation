from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import read_manifest
from .preprocess import load_gray, save_mask, save_overlay
from .pseudo import PseudoConfig, deterministic_pseudo


def _bbox_to_str(bbox) -> str:
    if bbox is None:
        return ""
    return ",".join(str(int(x)) for x in bbox)


def generate_pseudo(
    manifest: Path,
    out_dir: Path,
    cfg: PseudoConfig,
    max_pairs: int | None = None,
    overlays: int = 200,
    save_images: bool = True,
) -> Path:
    rows = read_manifest(manifest)
    if max_pairs:
        rows = rows[:max_pairs]
    masks_b = out_dir / "masks_B"
    masks_a = out_dir / "masks_A"
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_rows: list[dict[str, object]] = []
    for idx, row in enumerate(tqdm(rows, desc="pseudo")):
        pre = load_gray(Path(row["pre_path"]))
        post = load_gray(Path(row["post_path"]))
        result = deterministic_pseudo(pre, post, cfg)
        mask_name = f"{row['pair_id']}.png"
        mask_b_path = masks_b / mask_name
        mask_a_path = masks_a / mask_name
        if save_images:
            save_mask(result.mask_original, mask_b_path)
            save_mask(result.mask_original, mask_a_path)
        if save_images and idx < overlays:
            bbox = result.bbox_original
            save_overlay(post, result.mask_original, bbox, overlay_dir / f"{row['pair_id']}_B.png")
            save_overlay(pre, result.mask_original, bbox, overlay_dir / f"{row['pair_id']}_A.png")

        bbox = result.bbox_original
        pseudo_rows.append(
            {
                **row,
                "mask_B_path": str(mask_b_path),
                "mask_A_path": str(mask_a_path),
                "bbox_original": _bbox_to_str(bbox),
                "bbox_x0": bbox[0] if bbox else "",
                "bbox_y0": bbox[1] if bbox else "",
                "bbox_x1": bbox[2] if bbox else "",
                "bbox_y1": bbox[3] if bbox else "",
                **result.metrics(),
            }
        )

    out_csv = out_dir / "pseudo_manifest.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pseudo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pseudo_rows)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-percentile", type=float, default=96.5)
    parser.add_argument("--min-area-fraction", type=float, default=0.0010)
    parser.add_argument("--morph-radius", type=int, default=2)
    parser.add_argument("--pair-support-weight", type=float, default=0.20)
    parser.add_argument("--min-quality", type=float, default=0.14)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--target-height", type=int, default=496)
    parser.add_argument("--overlays", type=int, default=200)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    cfg = PseudoConfig(
        threshold_percentile=args.threshold_percentile,
        min_area_fraction=args.min_area_fraction,
        morph_radius=args.morph_radius,
        pair_support_weight=args.pair_support_weight,
        min_quality=args.min_quality,
        target_width=args.target_width,
        target_height=args.target_height,
    )
    out = generate_pseudo(args.manifest, args.out_dir, cfg, max_pairs=args.max_pairs, overlays=args.overlays, save_images=not args.no_images)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()






