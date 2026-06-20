from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from .make_pseudo import generate_pseudo
from .pseudo import PseudoConfig, config_asdict


def summarize_pseudo(pseudo_csv: Path) -> dict[str, float]:
    rows = list(csv.DictReader(pseudo_csv.open("r", encoding="utf-8-sig")))
    q = np.array([float(r["quality"]) for r in rows], dtype=np.float32)
    area = np.array([float(r["area_fraction"]) for r in rows], dtype=np.float32)
    comps = np.array([float(r["component_count"]) for r in rows], dtype=np.float32)
    edge = np.array([float(r["edge_touch_fraction"]) for r in rows], dtype=np.float32)
    nonempty = area > 0
    coverage = float(nonempty.mean())
    sane_area = float(((area[nonempty] >= 0.001) & (area[nonempty] <= 0.12)).mean()) if nonempty.any() else 0.0
    score = (
        0.40 * float(q.mean())
        + 0.22 * coverage
        + 0.18 * sane_area
        + 0.12 * float(np.exp(-abs(coverage - 0.55)))
        - 0.04 * float(np.mean(np.maximum(comps - 3, 0)))
        - 0.04 * float(edge.mean())
    )
    return {
        "selection_score": score,
        "quality_mean": float(q.mean()),
        "quality_median": float(np.median(q)),
        "coverage": coverage,
        "area_mean": float(area.mean()),
        "area_nonempty_mean": float(area[nonempty].mean()) if nonempty.any() else 0.0,
        "components_mean": float(comps.mean()),
        "edge_touch_mean": float(edge.mean()),
        "pairs": float(len(rows)),
    }


def write_report(out_dir: Path, best: dict[str, float | str], summaries: list[dict[str, float | str]]) -> None:
    lines = [
        "# Unsupervised Pseudo-Lesion Experiment",
        "",
        "This experiment uses only `数据集/1-30` paired OCT images.",
        "",
        "## Best Configuration",
        "",
    ]
    for key, value in best.items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Interpretation",
        "",
        "The selection score is label-free. It favors masks with strong postoperative saliency, reasonable non-empty coverage, sane area, few fragments, and low border contact.",
        "It is a quality-control heuristic, not clinical ground truth.",
        "",
        "The selected pseudo masks are in `best/pseudo_manifest.csv`, with transferred A masks under `best/masks_A`.",
        "",
        "## Top Runs",
        "",
        "| run | selection | coverage | quality | area_nonempty | components |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    top = sorted(summaries, key=lambda r: float(r["selection_score"]), reverse=True)[:8]
    for r in top:
        lines.append(
            f"| {r['run']} | {float(r['selection_score']):.4f} | {float(r['coverage']):.3f} | "
            f"{float(r['quality_mean']):.4f} | {float(r['area_nonempty_mean']):.4f} | {float(r['components_mean']):.2f} |"
        )
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "deterministic_grid")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--search-pairs", type=int, default=None, help="Default: search all pairs for research runs.")
    parser.add_argument("--overlays", type=int, default=200)
    args = parser.parse_args()

    thresholds = [95.5, 96.0, 96.5, 97.0, 97.5, 98.0]
    min_areas = [0.0005, 0.0010, 0.0015, 0.0020]
    radii = [1, 2, 3, 4]
    pair_weights = [0.00, 0.10, 0.20, 0.30]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | str]] = []
    for t in thresholds:
        for a in min_areas:
            for r in radii:
                for pw in pair_weights:
                    run = f"thr{t:g}_area{a:g}_r{r}_pair{pw:g}".replace(".", "p")
                    run_dir = args.out_dir / "runs" / run
                    cfg = PseudoConfig(
                        threshold_percentile=t,
                        min_area_fraction=a,
                        morph_radius=r,
                        pair_support_weight=pw,
                    )
                    pseudo_csv = generate_pseudo(
                        args.manifest,
                        run_dir,
                        cfg,
                        max_pairs=args.search_pairs,
                        overlays=0,
                        save_images=False,
                    )
                    summary = {"run": run, **config_asdict(cfg), **summarize_pseudo(pseudo_csv)}
                    summaries.append(summary)
                    print(run, summary["selection_score"], summary["coverage"])

    summary_csv = args.out_dir / "grid_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    best = max(summaries, key=lambda r: float(r["selection_score"]))
    best_src = args.out_dir / "runs" / str(best["run"])
    best_dst = args.out_dir / "best"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    cfg = PseudoConfig(
        threshold_percentile=float(best["threshold_percentile"]),
        min_area_fraction=float(best["min_area_fraction"]),
        morph_radius=int(best["morph_radius"]),
        pair_support_weight=float(best["pair_support_weight"]),
        min_quality=float(best["min_quality"]),
    )
    generate_pseudo(
        args.manifest,
        best_dst,
        cfg,
        max_pairs=args.max_pairs,
        overlays=args.overlays,
        save_images=True,
    )
    write_report(args.out_dir, best, summaries)
    print(f"Best: {best['run']}")
    print(f"Wrote {summary_csv}")
    print(f"Best pseudo manifest: {best_dst / 'pseudo_manifest.csv'}")


if __name__ == "__main__":
    main()





