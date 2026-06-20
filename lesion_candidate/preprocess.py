from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


Array = np.ndarray
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ResizeMeta:
    original_width: int
    original_height: int
    target_width: int
    target_height: int


def load_gray(path: Path) -> Array:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_mask(mask: Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_gray(arr: Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path)


def resize_image(arr: Array, size: tuple[int, int]) -> tuple[Array, ResizeMeta]:
    h, w = arr.shape[:2]
    tw, th = size
    out = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA if tw < w else cv2.INTER_LINEAR)
    return out.astype(np.float32), ResizeMeta(w, h, tw, th)


def robust_normalize(arr: Array, lower: float = 1.0, upper: float = 99.0) -> Array:
    lo, hi = np.percentile(arr, [lower, upper])
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def clahe(arr: Array, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Array:
    u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)).apply(u8)
    return eq.astype(np.float32) / 255.0


def retina_support(arr: Array, min_area_fraction: float = 0.02) -> Array:
    norm = robust_normalize(arr)
    thr = max(float(np.percentile(norm, 62)) * 0.35, 0.05)
    mask = norm > thr
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.ones_like(arr, dtype=bool)
    min_area = int(arr.size * min_area_fraction)
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == idx
    if not keep.any():
        keep = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return keep


def gradient_magnitude(arr: Array) -> Array:
    u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy) / 255.0


def local_zscore(arr: Array, sigma: float) -> Array:
    mean = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean2 = cv2.GaussianBlur(arr * arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 1e-6))
    return (arr - mean) / std


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def mask_to_bbox(mask: Array, min_area: int = 1) -> BBox | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < min_area:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_to_original(bbox: BBox | None, meta: ResizeMeta) -> BBox | None:
    if bbox is None:
        return None
    sx = meta.original_width / float(meta.target_width)
    sy = meta.original_height / float(meta.target_height)
    x0, y0, x1, y1 = bbox
    return clamp_bbox((x0 * sx, y0 * sy, x1 * sx, y1 * sy), meta.original_width, meta.original_height)


def mask_to_original(mask: Array, meta: ResizeMeta) -> Array:
    return cv2.resize(
        mask.astype(np.uint8),
        (meta.original_width, meta.original_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def draw_overlay(gray: Array, mask: Array, bbox: BBox | None = None) -> Array:
    base = np.repeat(np.clip(gray[..., None] * 255.0, 0, 255).astype(np.uint8), 3, axis=2)
    overlay = base.copy()
    overlay[mask.astype(bool)] = (255, 64, 64)
    out = cv2.addWeighted(base, 0.68, overlay, 0.32, 0)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    return out


def save_overlay(gray: Array, mask: Array, bbox: BBox | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_overlay(gray, mask, bbox)).save(path)


def combine_bboxes(boxes: Iterable[BBox]) -> BBox | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)
