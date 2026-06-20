from __future__ import annotations

import argparse
from pathlib import Path

from .data import assign_case_split, discover_pairs, manifest_summary, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "manifest_pairs_1_30.csv")
    parser.add_argument("--val-cases", type=int, default=6)
    args = parser.parse_args()

    records, bad_dirs = discover_pairs(args.dataset_root)
    records = assign_case_split(records, val_cases=args.val_cases)
    write_manifest(records, args.out)
    summary = manifest_summary(records)
    print(f"Wrote {args.out}")
    print(summary)
    print(f"Filtered non-standard folders: {len(bad_dirs)}")


if __name__ == "__main__":
    main()
