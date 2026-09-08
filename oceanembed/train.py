"""Training loop for OceanEmbed-Net.

Runs on CPU for a smoke test and on a GPU for real training, without any code
change. Typical use:

    # prove the machinery works, a couple of minutes on CPU
    python -m oceanembed.train --epochs 2 --samples-per-day 4 --width 16

    # real training, on Colab or any CUDA machine
    python -m oceanembed.train --epochs 60

Every run writes checkpoints and a metrics history to runs/<name>/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ROOT
from .dataset import OceanEmbedDataset
from .model import OceanEmbedNet, count_parameters, masked_depth_loss, profile_metrics


@dataclass
class TrainConfig:
    export_dir: str = "data/export"
    run_name: str = "baseline"
    epochs: int = 40
    batch_size: int = 16
    patch: int = 64
    samples_per_day: int = 16
    width: int = 48
    attn_blocks: int = 2
    heads: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 0
    amp: bool = True
    patience: int = 12


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, OceanEmbedDataset, OceanEmbedDataset]:
    export = Path(cfg.export_dir)
    if not export.is_absolute():
        export = ROOT / export

    train_ds = OceanEmbedDataset(
        export, "train", patch=cfg.patch,
        samples_per_day=cfg.samples_per_day, seed=cfg.seed,
    )
    # Validation uses whole domains, not random crops: the score should reflect
    # performance on the actual product, and it must not wander between epochs
    # because a different random window was drawn.
    val_ds = OceanEmbedDataset(export, "val", patch=None, seed=cfg.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=len(train_ds) > cfg.batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds, val_ds


def lr_at(epoch: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay.

    The warmup matters here because the output layer starts near zero, which
    is the climatological mean. A large first step would throw away that useful
    starting point before the network has learned anything to replace it with.
    """
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, device, depth_std: np.ndarray) -> dict:
    """Validation loss plus per-depth RMSE, bias and correlation in Celsius."""
    model.eval()
    total, n = 0.0, 0
    sums = None

    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        pred = model(x)
        total += float(masked_depth_loss(pred, y, v))
        n += 1

        m = profile_metrics(pred, y, v)
        if sums is None:
            sums = {k: val.clone() for k, val in m.items()}
        else:
            for k in sums:
                sums[k] += m[k]

    scale = torch.as_tensor(depth_std, device=device)
    out = {"loss": total / max(n, 1)}
    if sums is not None:
        # RMSE and bias are in standardised units; multiplying by each level's
        # standard deviation converts them to degrees Celsius. Correlation is
        # already dimensionless.
        out["rmse_c"] = (sums["rmse"] / max(n, 1) * scale).cpu().numpy()
        out["bias_c"] = (sums["bias"] / max(n, 1) * scale).cpu().numpy()
        out["corr"] = (sums["corr"] / max(n, 1)).cpu().numpy()
    return out


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    train_loader, val_loader, train_ds, val_ds = build_loaders(cfg)

    model = OceanEmbedNet(
        in_channels=train_ds.in_channels,
        out_depths=len(train_ds.depths),
        width=cfg.width,
        attn_blocks=cfg.attn_blocks,
        heads=cfg.heads,
    ).to(device)

    run_dir = ROOT / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"device        {device}")
    print(f"train         {train_ds.summary()}")
    print(f"val           {val_ds.summary()}")
    print(f"parameters    {count_parameters(model):,}")
    print(f"run           {run_dir}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    depth_std = train_ds.depth_std()
    history: list[dict] = []
    best = float("inf")
    stale = 0

    for epoch in range(cfg.epochs):
        lr = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        running, steps = 0.0, 0
        started = time.time()

        for x, y, v in train_loader:
            x, y, v = x.to(device), y.to(device), v.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = masked_depth_loss(model(x), y, v)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            steps += 1

        train_loss = running / max(steps, 1)
        val = evaluate(model, val_loader, device, depth_std)

        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "seconds": round(time.time() - started, 1),
        }
        if "rmse_c" in val:
            record["val_rmse_c_mean"] = float(np.mean(val["rmse_c"]))
            record["val_corr_mean"] = float(np.mean(val["corr"]))
            record["val_rmse_c_per_depth"] = [round(float(x), 4) for x in val["rmse_c"]]
            record["val_corr_per_depth"] = [round(float(x), 4) for x in val["corr"]]
        history.append(record)

        flag = ""
        if val["loss"] < best - 1e-5:
            best, stale = val["loss"], 0
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg),
                 "epoch": epoch, "val_loss": val["loss"],
                 "depths": train_ds.depths, "variables": train_ds.variables},
                run_dir / "best.pt",
            )
            flag = "  <- best"
        else:
            stale += 1

        line = (f"epoch {epoch:>3}  lr {lr:.2e}  train {train_loss:.4f}  "
                f"val {val['loss']:.4f}")
        if "rmse_c" in val:
            line += f"  rmse {record['val_rmse_c_mean']:.3f}C  corr {record['val_corr_mean']:.3f}"
        print(line + f"  [{record['seconds']}s]" + flag)

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if stale >= cfg.patience:
            print(f"\nno improvement for {cfg.patience} epochs, stopping early")
            break

    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run_dir / "last.pt")

    if history and "val_rmse_c_per_depth" in history[-1]:
        best_record = min(history, key=lambda r: r["val_loss"])
        print(f"\nbest epoch {best_record['epoch']}  "
              f"val loss {best_record['val_loss']:.4f}")
        print(f"\n{'depth':>8} {'RMSE (C)':>10} {'corr':>8}")
        for d, r, c in zip(train_ds.depths,
                           best_record["val_rmse_c_per_depth"],
                           best_record["val_corr_per_depth"]):
            print(f"{d:>7.0f}m {r:>10.3f} {c:>8.3f}")

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train OceanEmbed-Net.")
    d = TrainConfig()
    p.add_argument("--export-dir", default=d.export_dir)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--patch", type=int, default=d.patch)
    p.add_argument("--samples-per-day", type=int, default=d.samples_per_day)
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--attn-blocks", type=int, default=d.attn_blocks)
    p.add_argument("--heads", type=int, default=d.heads)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        export_dir=args.export_dir, run_name=args.run_name, epochs=args.epochs,
        batch_size=args.batch_size, patch=args.patch,
        samples_per_day=args.samples_per_day, width=args.width,
        attn_blocks=args.attn_blocks, heads=args.heads, lr=args.lr,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers, seed=args.seed, patience=args.patience,
        amp=not args.no_amp,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
