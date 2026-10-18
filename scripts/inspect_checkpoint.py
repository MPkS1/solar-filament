"""Inspect a saved training checkpoint (architecture + recorded validation score)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu")
    print(f"file: {args.checkpoint}")
    if not isinstance(obj, dict):
        print("  (raw state_dict, no metadata)")
        return
    print(f"  keys: {sorted(obj.keys())}")
    if "val_dice" in obj:
        print(f"  val_dice (recorded): {obj['val_dice']:.4f}")
    for k, v in (obj.get("args") or {}).items():
        print(f"  arg {k:<14} {v}")
    sd = obj.get("model", obj)
    if isinstance(sd, dict):
        n = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
        print(f"  params: {n / 1e6:.2f} M across {len(sd)} tensors")


if __name__ == "__main__":
    main()
