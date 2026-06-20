from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


FILENAME_RE = re.compile(
    r"(?P<patient>P(?P<case>\d{3}))_(?P<eye>OD|OS)_(?P<phase>pre|post)_vol(?P<vol>\d+)_slice(?P<slice>\d+)\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    case_index: str
    patient_id: str
    eye: str
    pair_folder: str
    pre_path: str
    post_path: str
    pre_name: str
    post_name: str
    pre_vol: int
    pre_slice: int
    post_vol: int
    post_slice: int
    same_vol: int
    same_slice: int
    width: int
    height: int
    split: str = "unassigned"


def reject_disallowed_dataset(dataset_root: Path) -> None:
    parts = {p.lower() for p in dataset_root.resolve().parts}
    if "1-60" in parts:
        raise ValueError("This project must not use dataset 1-60. Please pass dataset 1-30.")
    if dataset_root.name != "1-30":
        raise ValueError(f"Expected dataset root named '1-30', got: {dataset_root}")


def parse_oct_name(path: Path) -> dict[str, str | int]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected OCT filename: {path.name}")
    out: dict[str, str | int] = match.groupdict()
    out["vol"] = int(out["vol"])
    out["slice"] = int(out["slice"])
    return out


def discover_pairs(dataset_root: Path) -> tuple[list[PairRecord], list[Path]]:
    dataset_root = dataset_root.resolve()
    reject_disallowed_dataset(dataset_root)
    pairs: list[PairRecord] = []
    bad_dirs: list[Path] = []

    for case_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if not case_dir.name.isdigit():
            continue
        for pair_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            pngs = sorted(pair_dir.glob("*.png"))
            pre = [p for p in pngs if "_pre_" in p.name]
            post = [p for p in pngs if "_post_" in p.name]
            if len(pngs) != 2 or len(pre) != 1 or len(post) != 1:
                bad_dirs.append(pair_dir)
                continue

            pre_meta = parse_oct_name(pre[0])
            post_meta = parse_oct_name(post[0])
            with Image.open(pre[0]) as img:
                width, height = img.size
            case_index = case_dir.name
            pair_id = f"{case_index}_{pair_dir.name}_{pre[0].stem}_{post[0].stem}"
            safe_pair_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", pair_id)
            pairs.append(
                PairRecord(
                    pair_id=safe_pair_id,
                    case_index=case_index,
                    patient_id=str(pre_meta["patient"]),
                    eye=str(pre_meta["eye"]),
                    pair_folder=str(pair_dir.relative_to(dataset_root)),
                    pre_path=str(pre[0]),
                    post_path=str(post[0]),
                    pre_name=pre[0].name,
                    post_name=post[0].name,
                    pre_vol=int(pre_meta["vol"]),
                    pre_slice=int(pre_meta["slice"]),
                    post_vol=int(post_meta["vol"]),
                    post_slice=int(post_meta["slice"]),
                    same_vol=int(pre_meta["vol"] == post_meta["vol"]),
                    same_slice=int(pre_meta["slice"] == post_meta["slice"]),
                    width=width,
                    height=height,
                )
            )
    return pairs, bad_dirs


def assign_case_split(records: Sequence[PairRecord], val_cases: int = 6) -> list[PairRecord]:
    cases = sorted({r.case_index for r in records})
    val = set(cases[-val_cases:]) if val_cases > 0 else set()
    out: list[PairRecord] = []
    for r in records:
        split = "val" if r.case_index in val else "train"
        out.append(PairRecord(**{**asdict(r), "split": split}))
    return out


def write_manifest(records: Iterable[PairRecord], path: Path) -> None:
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def abs_path(value: str, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def manifest_summary(records: Sequence[PairRecord]) -> dict[str, int]:
    return {
        "pairs": len(records),
        "cases": len({r.case_index for r in records}),
        "same_vol": sum(r.same_vol for r in records),
        "same_slice": sum(r.same_slice for r in records),
        "train": sum(1 for r in records if r.split == "train"),
        "val": sum(1 for r in records if r.split == "val"),
    }

