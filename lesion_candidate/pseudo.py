from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .preprocess import (
    Array,
    BBox,
    bbox_to_original,
    clahe,
    gradient_magnitude,
    local_zscore,
    mask_to_bbox,
    mask_to_original,
    resize_image,
    retina_support,
    robust_normalize,
)


@dataclass(frozen=True)
class PseudoConfig:
    target_width: int = 768
    target_height: int = 496
    threshold_percentile: float = 96.5
    min_area_fraction: float = 0.0010
    max_area_fraction: float = 0.20
    morph_radius: int = 2
    pair_support_weight: float = 0.20
    dark_weight: float = 0.40
    bright_weight: float = 0.22
    contrast_weight: float = 0.22
    log_weight: float = 0.16
    min_quality: float = 0.14


@dataclass
class PseudoResult:
    mask_small: Array
    mask_original: Array
    score: Array
    bbox_small: BBox | None
    bbox_original: BBox | None
    quality: float
    mean_score: float
    max_score: float
    area_fraction: float
    component_count: int
    edge_touch_fraction: float
    empty_reason: str = ""

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "quality": self.quality,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "area_fraction": self.area_fraction,
            "component_count": self.component_count,
            "edge_touch_fraction": self.edge_touch_fraction,
            "empty_reason": self.empty_reason,
        }


def _normalize_score(score: Array) -> Array:
    score = np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(score, [2, 99.5])
    if hi <= lo + 1e-6:
        return np.zeros_like(score)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def multiscale_lesion_score(post: Array, pre: Array | None, cfg: PseudoConfig) -> tuple[Array, Array]:
    b = clahe(robust_normalize(post))
    support = retina_support(b)

    dark_scores: list[Array] = []
    bright_scores: list[Array] = []
    contrast_scores: list[Array] = []
    log_scores: list[Array] = []
    for sigma in (1.5, 2.5, 4.0, 6.0, 8.0, 12.0, 16.0):
        z = local_zscore(b, sigma=sigma)
        dark_scores.append(np.maximum(-z, 0.0))
        bright_scores.append(np.maximum(z, 0.0))
        blur = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)
        contrast_scores.append(np.abs(b - blur))
        log = cv2.Laplacian(cv2.GaussianBlur(b, (0, 0), sigmaX=sigma / 2.0), cv2.CV_32F, ksize=3)
        log_scores.append(np.abs(log))

    dark = _normalize_score(np.max(dark_scores, axis=0))
    bright = _normalize_score(np.max(bright_scores, axis=0))
    contrast = _normalize_score(np.max(contrast_scores, axis=0))
    log_score = _normalize_score(np.max(log_scores, axis=0))
    score = (
        cfg.dark_weight * dark
        + cfg.bright_weight * bright
        + cfg.contrast_weight * contrast
        + cfg.log_weight * log_score
    )

    if pre is not None and cfg.pair_support_weight > 0:
        a = clahe(robust_normalize(pre))
        pair_support = _normalize_score(0.55 * gradient_magnitude(a) + 0.45 * np.abs(a - cv2.GaussianBlur(a, (0, 0), 4)))
        score = (1.0 - cfg.pair_support_weight) * score + cfg.pair_support_weight * pair_support

    score = _normalize_score(score)
    score *= support.astype(np.float32)
    return score, support


def postprocess_score(score: Array, support: Array, cfg: PseudoConfig) -> tuple[Array, dict[str, float | int | str]]:
    h, w = score.shape
    min_area = max(8, int(h * w * cfg.min_area_fraction))
    max_area = int(h * w * cfg.max_area_fraction)
    active = score[support.astype(bool)]
    if active.size == 0:
        return np.zeros_like(score, dtype=bool), {"empty_reason": "no_retina_support"}

    thr = float(np.percentile(active, cfg.threshold_percentile))
    mask = (score >= thr) & support.astype(bool)
    radius = max(1, int(cfg.morph_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    kept = 0
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep |= labels == idx
            kept += 1
    mask = keep
    if not mask.any():
        return mask, {"empty_reason": "no_component_after_filter", "threshold": thr}
    return mask, {"empty_reason": "", "threshold": thr, "kept_components": kept}


def edge_touch_fraction(mask: Array) -> float:
    if not mask.any():
        return 0.0
    edge = np.zeros_like(mask, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return float(np.logical_and(mask, edge).sum() / max(mask.sum(), 1))


def quality_score(mask: Array, score: Array, support: Array) -> float:
    if not mask.any():
        return 0.0
    area_fraction = float(mask.mean())
    support_area = float(max(support.mean(), 1e-6))
    relative_area = area_fraction / support_area
    mean_score = float(score[mask].mean())
    components = max(1, cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    compact_penalty = min(1.0, 1.0 / components)
    area_penalty = float(np.exp(-abs(np.log(max(relative_area, 1e-6) / 0.035))))
    edge_penalty = 1.0 - min(edge_touch_fraction(mask), 0.5) * 2.0
    return float(mean_score * (0.55 + 0.45 * area_penalty) * (0.70 + 0.30 * compact_penalty) * edge_penalty)


def deterministic_pseudo(pre_original: Array, post_original: Array, cfg: PseudoConfig) -> PseudoResult:
    size = (cfg.target_width, cfg.target_height)
    pre, _ = resize_image(pre_original, size)
    post, meta = resize_image(post_original, size)
    score, support = multiscale_lesion_score(post, pre, cfg)
    mask, info = postprocess_score(score, support, cfg)
    quality = quality_score(mask, score, support)
    if quality < cfg.min_quality:
        mask = np.zeros_like(mask, dtype=bool)
        info["empty_reason"] = info.get("empty_reason") or "below_min_quality"

    bbox_small = mask_to_bbox(mask)
    mask_original = mask_to_original(mask, meta)
    bbox_original = bbox_to_original(bbox_small, meta)
    num = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1 if mask.any() else 0
    return PseudoResult(
        mask_small=mask,
        mask_original=mask_original,
        score=score,
        bbox_small=bbox_small,
        bbox_original=bbox_original,
        quality=quality,
        mean_score=float(score[mask].mean()) if mask.any() else 0.0,
        max_score=float(score.max()) if score.size else 0.0,
        area_fraction=float(mask.mean()),
        component_count=int(num),
        edge_touch_fraction=edge_touch_fraction(mask),
        empty_reason=str(info.get("empty_reason", "")),
    )


def config_asdict(cfg: PseudoConfig) -> dict[str, float | int]:
    return asdict(cfg)


