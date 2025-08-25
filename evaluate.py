from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from engine.validator import Validator
from engine.utils import BuildAutocastKWargs
from models.builder import build_model, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GroupDynamicsDataset(**cfg["dataset"]["val"])
    val_bs = cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    autocast_kwargs = BuildAutocastKWargs(cfg, device)
    validator = Validator(device=device, autocast_kwargs=autocast_kwargs)

    metrics, _ = validator.run(model, loader)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
