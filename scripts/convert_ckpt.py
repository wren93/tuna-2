#!/usr/bin/env python3
"""Convert Tuna-2 checkpoints: rename showo.model -> tuna.model, drop mask_token."""

import argparse
from pathlib import Path
import torch


def convert(src: Path, dst: Path) -> None:
    state = torch.load(src, map_location="cpu", weights_only=True)

    # Unwrap common checkpoint wrappers
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]

    converted = {}
    skipped = []
    for k, v in state.items():
        if k == "mask_token" or "vae" in k:
            skipped.append(k)
            continue
        new_k = k.replace("showo.", "tuna.", 1) if k.startswith("showo.") else k
        converted[new_k] = v

    if skipped:
        print(f"Skipped: {skipped}")
    renamed = sum(1 for k in state if k.startswith("showo."))
    print(f"Renamed {renamed} keys  |  kept {len(converted) - (len(state) - renamed - len(skipped))} keys unchanged")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    print(f"Saved -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source checkpoint (.pt)")
    parser.add_argument("dst", type=Path, help="Destination checkpoint (.pt)")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
