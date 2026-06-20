from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import read_manifest
from .models import LatentCritic, MUISStyleEncoder, MutualInfoDiscriminator
from .preprocess import load_gray, resize_image, robust_normalize


class PostDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int = 192, aux_size: int = 384) -> None:
        self.rows = rows
        self.image_size = image_size
        self.aux_size = aux_size

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str, size: int) -> np.ndarray:
        img = load_gray(Path(path))
        img = resize_image(img, (size, size))
        img = robust_normalize(img)
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = self._load(row["post_path"], self.image_size)
        x2 = self._load(row["post_path"], self.aux_size)
        x_aug = x.copy()
        if np.random.rand() < 0.5:
            x_aug = np.ascontiguousarray(x_aug[:, ::-1])
        if np.random.rand() < 0.8:
            gamma = np.random.uniform(0.75, 1.35)
            gain = np.random.uniform(0.85, 1.20)
            bias = np.random.uniform(-0.08, 0.08)
            x_aug = np.clip((x_aug ** gamma) * gain + bias, 0.0, 1.0)
        if np.random.rand() < 0.5:
            x_aug = np.clip(x_aug + np.random.normal(0.0, np.random.uniform(0.005, 0.025), x_aug.shape), 0.0, 1.0)
        return (
            torch.from_numpy(x[None].astype(np.float32)),
            torch.from_numpy(x_aug[None].astype(np.float32)),
            torch.from_numpy(x2[None].astype(np.float32)),
            row["id"],
        )


def sample_prior(batch: int, dim_zs: int, dim_zc: int, device: torch.device) -> torch.Tensor:
    zs = torch.randn(batch, dim_zs, device=device) * 0.1
    ids = torch.randint(0, dim_zc, (batch,), device=device)
    zc = F.one_hot(ids, dim_zc).float()
    return torch.cat([zs, zc], dim=1)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, device=real.device)
    inter = eps * real + (1.0 - eps) * fake
    inter.requires_grad_(True)
    pred = critic(inter)
    grad = torch.autograd.grad(pred.sum(), inter, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean() * 10.0


def train_teacher(
    manifest: Path,
    out_dir: Path,
    dim_zc: int = 8,
    dim_zs: int = 64,
    epochs: int = 300,
    batch_size: int = 64,
    image_size: int = 192,
    aux_size: int = 384,
    lr: float = 2e-4,
    base: int = 64,
    num_workers: int = 8,
    critic_steps: int = 5,
    use_amp: bool = True,
    compile_model: bool = False,
) -> Path:
    rows = [r for r in read_manifest(manifest) if r.get("split", "train") == "train"]
    if not rows:
        raise ValueError(f"No training rows found in {manifest}")

    loader = DataLoader(
        PostDataset(rows, image_size=image_size, aux_size=aux_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")
    encoder = MUISStyleEncoder(dim_zs=dim_zs, dim_zc=dim_zc, base=base).to(device)
    critic = LatentCritic(dim_zs=dim_zs, dim_zc=dim_zc).to(device)
    disc = MutualInfoDiscriminator(feature_dim=encoder.feature_dim, z_dim=dim_zs + dim_zc).to(device)
    if compile_model and hasattr(torch, "compile"):
        encoder = torch.compile(encoder)  # type: ignore[assignment]
        critic = torch.compile(critic)  # type: ignore[assignment]
        disc = torch.compile(disc)  # type: ignore[assignment]

    opt_e = torch.optim.AdamW(list(encoder.parameters()) + list(disc.parameters()), lr=lr, weight_decay=1e-4)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)
    sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=max(1, epochs), eta_min=lr * 0.05)
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=max(1, epochs), eta_min=lr * 0.05)
    scaler_e = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_c = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    l1 = nn.L1Loss()
    logs: list[dict[str, float | int]] = []
    step = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        encoder.train()
        critic.train()
        disc.train()
        meters = {"critic": [], "enc": [], "mi": [], "aug": [], "scale": []}
        for x, x_aug, x2, _ in tqdm(loader, desc=f"teacher {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            x_aug = x_aug.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            batch = x.size(0)
            train_critic = step % max(1, critic_steps) != max(1, critic_steps) - 1
            if train_critic:
                with torch.no_grad():
                    zs, zc_logits, _, _, _ = encoder(x)
                    z = torch.cat([zs, F.softmax(zc_logits, dim=1)], dim=1)
                z_prior = sample_prior(batch, dim_zs, dim_zc, device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss_c = critic(z).mean() - critic(z_prior).mean() + gradient_penalty(critic, z_prior, z)
                opt_c.zero_grad(set_to_none=True)
                scaler_c.scale(loss_c).backward()
                scaler_c.unscale_(opt_c)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                scaler_c.step(opt_c)
                scaler_c.update()
                meters["critic"].append(float(loss_c.item()))
            else:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    zs, zc_logits, bottle, _, _ = encoder(x)
                    zc = F.softmax(zc_logits, dim=1)
                    z = torch.cat([zs, zc], dim=1)
                    _, zc_aug, _, _, _ = encoder(x_aug)
                    adv = -critic(z).mean()
                    labels = torch.zeros(batch * 2, 1, device=device)
                    labels[:batch] = 1
                    shuffled = z[torch.randperm(batch)]
                    mi_logits = disc(torch.cat([bottle, bottle], dim=0), torch.cat([z, shuffled], dim=0))
                    mi = bce(mi_logits, labels)
                    aug = kl(F.log_softmax(zc_aug, dim=1), zc.detach())
                    x2_small = F.interpolate(x2, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    _, _, bottle2, _, _ = encoder(x2_small)
                    scale = l1(bottle, bottle2)
                    loss_e = adv + 0.5 * mi + 6.0 * aug + 5.0 * scale
                opt_e.zero_grad(set_to_none=True)
                scaler_e.scale(loss_e).backward()
                scaler_e.unscale_(opt_e)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(disc.parameters()), 5.0)
                scaler_e.step(opt_e)
                scaler_e.update()
                meters["enc"].append(float(loss_e.item()))
                meters["mi"].append(float(mi.item()))
                meters["aug"].append(float(aug.item()))
                meters["scale"].append(float(scale.item()))
            step += 1
        sched_e.step()
        sched_c.step()
        row = {
            "epoch": epoch,
            "lr": float(sched_e.get_last_lr()[0]),
            **{k: float(np.mean(v)) if v else 0.0 for k, v in meters.items()},
        }
        logs.append(row)
        print(row)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "dim_zs": dim_zs,
                "dim_zc": dim_zc,
                "image_size": image_size,
                "aux_size": aux_size,
                "base": base,
                "critic_steps": critic_steps,
                "architecture": "muis_residual_encoder",
            },
            out_dir / "teacher_last.pt",
        )
    with (out_dir / "teacher_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)
    return out_dir / "teacher_last.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "teacher")
    parser.add_argument("--dim-zc", type=int, default=8)
    parser.add_argument("--dim-zs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--aux-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()
    ckpt = train_teacher(
        args.manifest,
        args.out_dir,
        dim_zc=args.dim_zc,
        dim_zs=args.dim_zs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        aux_size=args.aux_size,
        lr=args.lr,
        base=args.base,
        num_workers=args.num_workers,
        critic_steps=args.critic_steps,
        use_amp=not args.no_amp,
        compile_model=args.compile_model,
    )
    print(f"Wrote {ckpt}")


if __name__ == "__main__":
    main()
