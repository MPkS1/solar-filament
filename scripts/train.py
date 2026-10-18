"""Train the U-Net filament segmenter.

Examples
--------
Smoke test on CPU (a handful of crops, one epoch):
    python scripts/train.py --smoke

Full run (Kaggle GPU):
    python scripts/train.py --epochs 40 --batch 16 --crop 512 --base 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magfilo.data import IMAGENET_LIKE, collate, split_stems, views_for_stems, FilamentDataset  # noqa: E402
from magfilo.coco import load_coco  # noqa: E402
from magfilo.model import UNet, combined_loss  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, device: str, max_batches: int = 20) -> tuple[float, float]:
    model.eval()
    dices, ipos = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        prob = torch.sigmoid(model(x))
        pred = (prob > 0.5).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1.0) / (denom + 1.0)
        dices.extend(dice.cpu().tolist())
        ipos.append(float(y.mean().item()))
    return float(np.mean(dices)), float(np.mean(ipos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap views for a quick run")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="checkpoints/unet.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the plumbing")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.crop, args.base = 1, 2, 256, 8
        args.limit = args.limit or 6
        args.workers = 0

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={device} epochs={args.epochs} crop={args.crop} base={args.base} limit={args.limit}")

    coco = load_coco()
    train_stems, val_stems = split_stems(coco, args.val_frac, args.seed)
    print(f"splits by stem: {len(train_stems)} train / {len(val_stems)} val physical images")

    train_ds = FilamentDataset(views_for_stems(coco, train_stems), crop=args.crop, train=True, limit=args.limit)
    val_ds = FilamentDataset(views_for_stems(coco, val_stems), crop=args.crop, train=False,
                             pos_crop_prob=1.0, limit=args.limit)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate)
    print(f"train crops/epoch: {len(train_ds)}  val crops: {len(val_ds)}")

    model = UNet(in_ch=1, base=args.base, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for batch in train_ld:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            loss = combined_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            nb += 1
        sched.step()
        val_dice, val_pos = evaluate(model, val_ld, device)
        print(f"epoch {epoch:>3}  loss {running / max(nb, 1):.4f}  "
              f"val_dice {val_dice:.4f}  val_pos_frac {val_pos:.4f}  ({time.time() - t0:.0f}s)")
        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "args": vars(args), "val_dice": best}, out)
            print(f"  saved {out} (val_dice {best:.4f})")

    print(f"done. best val_dice {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
