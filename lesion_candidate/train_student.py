from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models import (
    StudentUNet,
    dice_loss_with_logits,
    focal_loss_with_logits,
    main_logits,
    tversky_loss_with_logits,
)
from .preprocess import load_gray, resize_image, robust_normalize, save_overlay


class StudentDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: tuple[int, int], augment: bool = False) -> None:
        self.rows = rows
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.20:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        if random.random() < 0.80:
            gamma = random.uniform(0.75, 1.35)
            gain = random.uniform(0.85, 1.20)
            bias = random.uniform(-0.08, 0.08)
            image = np.clip((np.clip(image, 0.0, 1.0) ** gamma) * gain + bias, 0.0, 1.0)
        if random.random() < 0.50:
            image = np.clip(image + np.random.normal(0.0, random.uniform(0.005, 0.030), image.shape).astype(np.float32), 0.0, 1.0)
        if random.random() < 0.35:
            h, w = image.shape
            angle = random.uniform(-4.0, 4.0)
            scale = random.uniform(0.96, 1.04)
            tx = random.uniform(-0.025, 0.025) * w
            ty = random.uniform(-0.025, 0.025) * h
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            mat[:, 2] += (tx, ty)
            image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pre = robust_normalize(load_gray(Path(row["pre_path"])))
        mask = load_gray(Path(row["mask_A_path"])) > 0.5
        pre, _ = resize_image(pre, self.image_size)
        mask = cv2.resize(mask.astype(np.uint8), self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.augment:
            pre, mask = self._augment(pre, mask)
        x = torch.from_numpy(pre[None].astype(np.float32))
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y, row["pair_id"]


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = [r for r in rows if r.get("split", "train") == "train"]
    val = [r for r in rows if r.get("split", "train") == "val"]
    if not val:
        cases = sorted({r["case_index"] for r in rows})
        val_cases = set(cases[-max(1, len(cases) // 5) :])
        train = [r for r in rows if r["case_index"] not in val_cases]
        val = [r for r in rows if r["case_index"] in val_cases]
    return train, val


def parse_thresholds(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def compute_auto_pos_weight(rows: list[dict[str, str]], max_value: float = 120.0) -> float:
    positive = 0.0
    total = 0.0
    for row in rows:
        mask = load_gray(Path(row["mask_A_path"])) > 0.5
        positive += float(mask.sum())
        total += float(mask.size)
    fraction = positive / max(total, 1.0)
    if fraction <= 0:
        return max_value
    return float(min(max_value, max(1.0, (1.0 - fraction) / fraction)))


def batch_iou_at_threshold(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    pred = torch.sigmoid(logits) > threshold
    target = target > 0.5
    inter = torch.logical_and(pred, target).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(pred, target).sum(dim=(1, 2, 3)).float()
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(iou.mean().item())


def segmentation_loss(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    outputs = output if isinstance(output, tuple) else (output,)
    weights = (1.0, 0.40, 0.25, 0.15)
    total = target.new_tensor(0.0)
    norm = 0.0
    for idx, logits in enumerate(outputs):
        weight = weights[min(idx, len(weights) - 1)]
        loss = (
            focal_loss_with_logits(logits, target, alpha=0.75, gamma=2.0, pos_weight=pos_weight)
            + dice_loss_with_logits(logits, target)
            + tversky_loss_with_logits(logits, target, alpha=0.30, beta=0.70)
        )
        total = total + weight * loss
        norm += weight
    return total / norm


def evaluate(
    model: StudentUNet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    thresholds: list[float],
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    ious = {thr: [] for thr in thresholds}
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = main_logits(model(x))
                loss = segmentation_loss(logits, y, pos_weight)
            losses.append(float(loss.item()))
            for thr in thresholds:
                ious[thr].append(batch_iou_at_threshold(logits.float(), y, thr))
    means = {thr: float(np.mean(values)) if values else 0.0 for thr, values in ious.items()}
    best_thr = max(means, key=means.get) if means else 0.5
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_best_iou": float(means.get(best_thr, 0.0)),
        "val_best_threshold": float(best_thr),
        **{f"val_iou_t{thr:g}".replace(".", "p"): value for thr, value in means.items()},
    }


def train(
    manifest: Path,
    out_dir: Path,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 2e-4,
    image_size: tuple[int, int] = (768, 496),
    base: int = 64,
    dropout: float = 0.10,
    max_train: int | None = None,
    max_val: int | None = None,
    pos_weight: str = "auto",
    num_workers: int = 4,
    weight_decay: float = 1e-4,
    consistency_weight: float = 0.20,
    thresholds: list[float] | None = None,
    use_amp: bool = True,
    deep_supervision: bool = True,
    compile_model: bool = False,
) -> Path:
    rows = [r for r in read_rows(manifest) if Path(r["mask_A_path"]).exists()]
    train_rows, val_rows = split_rows(rows)
    if max_train:
        train_rows = train_rows[:max_train]
    if max_val:
        val_rows = val_rows[:max_val]
    if not train_rows:
        raise ValueError("No training rows with mask_A_path were found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = thresholds or [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    resolved_pos_weight = compute_auto_pos_weight(train_rows) if pos_weight == "auto" else float(pos_weight)

    train_loader = DataLoader(
        StudentDataset(train_rows, image_size, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        StudentDataset(val_rows, image_size, augment=False),
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")
    model = StudentUNet(base=base, dropout=dropout, deep_supervision=deep_supervision).to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight_tensor = torch.tensor(float(resolved_pos_weight), device=device)

    log_rows: list[dict[str, float | int]] = []
    best_iou = -1.0
    best_path = out_dir / "student_best.pt"
    last_path = out_dir / "student_last.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, _ in tqdm(train_loader, desc=f"train {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(x)
                logits = main_logits(output)
                loss = segmentation_loss(output, y, pos_weight_tensor)
                if consistency_weight > 0:
                    scale = random.choice((0.75, 1.25))
                    x_scaled = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
                    pred_scaled = main_logits(model(x_scaled))
                    pred_scaled = torch.nn.functional.interpolate(pred_scaled, size=logits.shape[-2:], mode="bilinear", align_corners=False)
                    consistency = torch.mean(torch.abs(torch.sigmoid(logits.detach()) - torch.sigmoid(pred_scaled)))
                    loss = loss + consistency_weight * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        scheduler.step()
        metrics = evaluate(model, val_loader, device, pos_weight_tensor, thresholds, use_amp) if val_rows else {"val_loss": 0.0, "val_best_iou": 0.0, "val_best_threshold": 0.5}
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "lr": float(scheduler.get_last_lr()[0]),
            "pos_weight": float(resolved_pos_weight),
            **metrics,
        }
        log_rows.append(row)
        print(row)
        checkpoint = {
            "model": (model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()),
            "image_size": image_size,
            "base": base,
            "dropout": dropout,
            "deep_supervision": deep_supervision,
            "architecture": "residual_attention_aspp_unet",
            "pos_weight": float(resolved_pos_weight),
            "threshold": float(metrics.get("val_best_threshold", 0.5)),
            "epoch": epoch,
            "val_best_iou": float(metrics.get("val_best_iou", 0.0)),
        }
        torch.save(checkpoint, last_path)
        if float(metrics.get("val_best_iou", 0.0)) >= best_iou:
            best_iou = float(metrics.get("val_best_iou", 0.0))
            torch.save(checkpoint, best_path)

    if log_rows:
        with (out_dir / "train_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[-1].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        save_val_predictions(model, val_rows[:64], out_dir / "val_predictions", image_size, device, threshold=float(log_rows[-1].get("val_best_threshold", 0.5)))
        write_report(out_dir, len(train_rows), len(val_rows), log_rows[-1], best_iou, best_path)
    return best_path


def save_val_predictions(
    model: StudentUNet,
    rows: list[dict[str, str]],
    out_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
    threshold: float,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for row in rows:
            pre_original = robust_normalize(load_gray(Path(row["pre_path"])))
            pre, meta = resize_image(pre_original, image_size)
            x = torch.from_numpy(pre[None, None].astype(np.float32)).to(device)
            pred = torch.sigmoid(main_logits(model(x)))[0, 0].cpu().numpy() > threshold
            pred_original = cv2.resize(pred.astype(np.uint8), (meta.original_width, meta.original_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            save_overlay(pre_original, pred_original, None, out_dir / f"{row['pair_id']}.png")


def write_report(out_dir: Path, n_train: int, n_val: int, last: dict[str, float | int], best_iou: float, best_path: Path) -> None:
    lines = [
        "# Student Training Report",
        "",
        f"- Train pseudo-labeled pairs: {n_train}",
        f"- Validation pseudo-labeled pairs: {n_val}",
        f"- Last train loss: {float(last['train_loss']):.4f}",
        f"- Last validation loss: {float(last['val_loss']):.4f}",
        f"- Best validation pseudo IoU: {best_iou:.4f}",
        f"- Last selected threshold: {float(last['val_best_threshold']):.3f}",
        f"- Best checkpoint: `{best_path.name}`",
        "",
        "Validation IoU is measured against pseudo masks, not manual labels.",
    ]
    (out_dir / "student_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "student_research")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=496)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.20)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()
    ckpt = train(
        args.manifest,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=(args.width, args.height),
        base=args.base,
        dropout=args.dropout,
        max_train=args.max_train,
        max_val=args.max_val,
        pos_weight=str(args.pos_weight),
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        consistency_weight=args.consistency_weight,
        thresholds=parse_thresholds(args.thresholds),
        use_amp=not args.no_amp,
        deep_supervision=not args.no_deep_supervision,
        compile_model=args.compile_model,
    )
    print(f"Wrote {ckpt}")


if __name__ == "__main__":
    main()
