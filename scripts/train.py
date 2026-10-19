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
from magfilo.disk import disk_mask  # noqa: E402
from magfilo.metrics import pq_pooled  # noqa: E402
from magfilo.model import UNet, combined_loss  # noqa: E402
from magfilo.predict import predict_logits_tiled, prob_to_instances  # noqa: E402
from magfilo.runtime import pick_device  # noqa: E402
from magfilo.data import read_image  # noqa: E402
from magfilo import paths  # noqa: E402


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


@torch.no_grad()
def evaluate_pq(model: UNet, coco, stems: list[str], device: str, tile: int,
                threshold: float, min_area: int, preprocess: bool, tta: bool,
                relative_disk_radius: float = 0.98) -> dict:
    """Official pooled PQ on full held-out frames.

    The old ``val_dice`` was measured on filament-centred crops only, so it never
    saw background and could not see the false-positive rate at all -- which is
    why the held-out estimate and the leaderboard disagreed. This scores exactly
    what the organizers score, on whole frames.
    """
    from magfilo.data import rasterize_view_instances
    from magfilo.preprocess import preprocess_frame

    entries = []
    n_preds = []
    for stem in stems:
        gray = read_image(paths.TRAIN_IMAGES / f"{stem}.jpeg")
        if gray is None:
            continue
        src = preprocess_frame(gray)[0] if preprocess else gray
        logits = predict_logits_tiled(model, src, device=device, tile=tile,
                                      preprocess=preprocess, tta=tta)
        prob = 1.0 / (1.0 + np.exp(-logits))
        labels = prob_to_instances(prob, threshold=threshold, min_area=min_area,
                                   disk=disk_mask(gray, relative_radius=relative_disk_radius))
        n_preds.append(int(labels.max()))
        for view in coco.by_stem[stem]:
            entries.append((rasterize_view_instances(view, 2048, 1.0).astype(np.int32), labels))
    rep = pq_pooled(entries)
    return {"pq": rep.pq, "tp": rep.tp, "fp": rep.fp, "fn": rep.fn,
            "sem_dice": rep.semantic_dice_mean, "n_pred": float(np.mean(n_preds)) if n_preds else 0.0}


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
    ap.add_argument("--preprocess", action="store_true",
                    help="train on limb-flattened, local-contrast input (must match inference)")
    ap.add_argument("--sigma", type=float, default=25.0, help="local-background scale for --preprocess")
    ap.add_argument("--val-pq-frames", type=int, default=0,
                    help="score this many whole held-out frames with the official PQ each epoch "
                         "and select the checkpoint on it (needs a GPU; 0 = use crop Dice)")
    ap.add_argument("--val-pq-threshold", type=float, default=0.5)
    ap.add_argument("--val-pq-min-area", type=int, default=40)
    ap.add_argument("--val-pq-tta", action="store_true")
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

    train_ds = FilamentDataset(views_for_stems(coco, train_stems), crop=args.crop, train=True,
                               limit=args.limit, preprocess=args.preprocess, sigma=args.sigma)
    val_ds = FilamentDataset(views_for_stems(coco, val_stems), crop=args.crop, train=False,
                             pos_crop_prob=1.0, limit=args.limit, preprocess=args.preprocess,
                             sigma=args.sigma)
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
        msg = (f"epoch {epoch:>3}  loss {running / max(nb, 1):.4f}  "
               f"val_dice {val_dice:.4f}  val_pos_frac {val_pos:.4f}")
        score = val_dice
        if args.val_pq_frames > 0:
            pq = evaluate_pq(model, coco, val_stems[:args.val_pq_frames], device,
                             tile=args.crop, threshold=args.val_pq_threshold,
                             min_area=args.val_pq_min_area, preprocess=args.preprocess,
                             tta=args.val_pq_tta)
            msg += (f"  val_PQ {pq['pq']:.4f} (TP/FP/FN {pq['tp']}/{pq['fp']}/{pq['fn']}, "
                    f"n_pred {pq['n_pred']:.1f})")
            score = pq["pq"]
        print(msg + f"  ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_dice": val_dice, "val_score": best,
                        "selection": "pq" if args.val_pq_frames > 0 else "crop_dice"}, out)
            print(f"  saved {out} (selection score {best:.4f})")

    print(f"done. best selection score {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
