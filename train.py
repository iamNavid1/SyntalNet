from __future__ import annotations

import re
import os
import yaml
import glob
import math
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader, DistributedSampler

from data.dataset import GroupDynamicsDataset
from data.collate import collate_fn
from data.transforms import StandardizeTransform
from engine.trainer import Trainer
from models.builder import build_model, load_config
from utils.logger import setup_logger
from utils.scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.model_stats import *


# ----------------------------- arg parsing -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None, help="Specific checkpoint path to resume")
    parser.add_argument("--logo", action="store_true", help="Leave-One-Group-Out cross validation (flag only; wire up in your Trainer)")
    return parser.parse_args()

# ----------------------------- distributed utils -----------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def init_distributed_mode() -> Tuple[bool, int, int, int, torch.device]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = env_world_size
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else None

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    return distributed, rank, world_size, local_rank, device

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

# ----------------------------- reproducibility -----------------------------

def set_seed(seed: int, add_rank: bool = True):
    s = seed + get_rank() if (add_rank and is_dist_avail_and_initialized()) else seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

# ----------------------------- IO helpers -----------------------------

def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    pattern = os.path.join(ckpt_dir, "epoch_*.pth")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None

def discover_group_ids(root_dir: str, modalities: List[str]) -> List[int]:
    example_mod = modalities[0]
    pattern_csv = os.path.join(root_dir, example_mod, "Group_*.csv")
    pattern_json = os.path.join(root_dir, example_mod, "Group_*.json")
    files = glob.glob(pattern_csv) + glob.glob(pattern_json)
    group_ids = set()
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"Group_(\d+)\.(csv|json)$", base)
        if m:
            group_ids.add(int(m.group(1)))
    return sorted(group_ids)

# ----------------------------- Builders -----------------------------

def build_datasets(cfg, logo_held_out=None):
    args = cfg["dataset"]["args"]
    train_cfg = {**args}
    val_cfg   = {**args}

    norm_stats = cfg["dataset"].get("stats_dir")
    if isinstance(norm_stats, str):
        with open(norm_stats, "r") as f:
            norm_stats = yaml.safe_load(f)
    if norm_stats:
        transform = StandardizeTransform(norm_stats)
        train_cfg["transforms"] = transform
        val_cfg["transforms"] = transform
        args["transforms"] = transform

    if logo_held_out is not None:
        train_cfg["exclude_groups"] = [logo_held_out]
        val_cfg["include_groups"] = [logo_held_out]
        return GroupDynamicsDataset(**train_cfg), GroupDynamicsDataset(**val_cfg)
    
    split_cfg = cfg["dataset"].get("split")
    mode = split_cfg.get("mode", "item")
    split_seed = int(split_cfg.get("seed", cfg["training"].get("seed", 42)))

    if mode == "item":
        full = GroupDynamicsDataset(**args)
        n = len(full)
        n_val = int(round(n * float(split_cfg.get("ratio", 0.2))))
        n_train = n - n_val
        g = torch.Generator().manual_seed(split_seed)
        train_subset, val_subset = random_split(full, [n_train, n_val], generator=g)
        return train_subset, val_subset

    elif mode == "group":
        all_groups = discover_group_ids(args["root_dir"], args["modalities"])
        val_groups = split_cfg.get("groups")
        if not val_groups:
            rng = random.Random(split_seed)
            k = max(1, int(round(len(all_groups) * float(split_cfg.get("ratio", 0.2)))))
            val_groups = sorted(rng.sample(all_groups, k))
        train_cfg["exclude_groups"] = val_groups
        val_cfg["include_groups"]   = val_groups
        return GroupDynamicsDataset(**train_cfg), GroupDynamicsDataset(**val_cfg)

    else:
        raise ValueError(f"Unknown split.mode: {mode}")

def build_loaders(cfg, train_dataset, val_dataset, distributed: bool):
    num_workers = cfg["training"].get("num_workers", 4)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        dataset            = train_dataset,
        batch_size         = cfg["training"]["batch_size"],
        sampler            = train_sampler,
        shuffle            = (not distributed),
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
        collate_fn         = collate_fn,
        worker_init_fn     = _worker_init_fn,
        generator          = torch.Generator().manual_seed(0 if not is_dist_avail_and_initialized() else get_rank()),
    )
    val_loader = DataLoader(
        dataset            = val_dataset,
        batch_size         = cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
        sampler            = val_sampler,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
        collate_fn         = collate_fn,
        worker_init_fn     = _worker_init_fn,
        generator          = torch.Generator().manual_seed(1 if not is_dist_avail_and_initialized() else get_rank() + 1),
    )
    return train_loader, val_loader

# ----------------------------- Single-run training -----------------------------

def run_training(cfg, args, device, local_rank, distributed, logger, writer):
    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset, distributed)

    model = build_model(cfg).to(device)

    if is_main_process() and logger:
        log_training_hyperparams(cfg, logger)
        log_model_hyperparams(cfg, logger)
        log_param_counts(model, logger)
    
    if distributed:
        if device.type != "cuda":
            raise RuntimeError("CPU distributed not supported; use CUDA or run single-process.")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = build_optimizer(model, cfg)

    accum = max(1, cfg["training"].get("accum_steps", 1))
    total_steps = math.ceil(len(train_loader) / accum) * cfg["training"]["epochs"]
    scheduler_cfg = {
        "warmup_steps" : int(cfg["training"].get("warmup_ratio", 0) * total_steps),
        "max_steps"    : total_steps,
        "min_lr"       : cfg["training"].get("min_lr", 0.0),
        "type"         : cfg["training"].get("scheduler", "warmup_cosine"),
    }
    scheduler = build_scheduler(optimizer, scheduler_cfg)


    trainer = Trainer(
        cfg          = cfg,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
        logger       = logger,
        writer       = writer,
        world_size   = get_world_size(),
    )

    ckpt_dir = cfg["logging"]["checkpoint_dir"]
    if args.resume:
        resume_path = args.resume_path or find_latest_checkpoint(ckpt_dir)
        if resume_path is None:
            if is_main_process() and logger:
                logger.warning("Resume requested but no checkpoint found in %s", ckpt_dir)
        else:
            trainer.resume_from(resume_path)

    trainer.train(
        epochs=cfg["training"]["epochs"],
        ckpt_dir=ckpt_dir,
        validate_interval=cfg["training"].get("val_interval", 1),
        checkpoint_interval=cfg["training"].get("checkpoint_interval", 1),
    )

# ----------------------------- LOGO helpers -----------------------------

def fold_dirs(base_log_dir: str, base_ckpt_dir: str, held_out_group: int) -> Tuple[str, str]:
    tag = f"fold_{held_out_group:02d}"
    return (os.path.join(base_log_dir,  "logo", tag),
            os.path.join(base_ckpt_dir, "logo", tag))

DONE_MARK = "_FOLD_DONE.json"

def mark_fold_done(ckpt_dir: str, meta: dict):
    if is_main_process():
        path = os.path.join(ckpt_dir, DONE_MARK)
        with open(path, "w") as f:
            json.dump(meta, f)

def fold_is_done(ckpt_dir: str) -> bool:
    return os.path.isfile(os.path.join(ckpt_dir, DONE_MARK))

def parse_held_out_from_path(path: str) -> Optional[int]:
    parts = Path(path).parts
    for p in parts:
        m = re.match(r"fold_(\d+)", p)
        if m:
            return int(m.group(1))
    return None

def find_resume_state_for_logo(base_ckpt_dir: str, all_groups: List[int], explicit_path: Optional[str]) -> Tuple[int, Optional[str]]:
    if explicit_path is not None:
        gid = parse_held_out_from_path(explicit_path)
        if gid is None:
            raise ValueError(f"--resume-path does not include a fold_XX segment: {explicit_path}")
        try:
            start_idx = all_groups.index(gid)
        except ValueError:
            raise ValueError(f"Held-out group {gid} (from resume path) is not in discovered groups: {all_groups}")
        return start_idx, explicit_path

    for i, gid in enumerate(all_groups):
        _, cdir = fold_dirs("", base_ckpt_dir, gid)
        if fold_is_done(cdir):
            continue
        latest = find_latest_checkpoint(cdir)
        return i, latest

    return len(all_groups), None  # all done

# ----------------------------- LOGO CV -----------------------------

def run_logo_cv(cfg, args, device, local_rank, distributed, base_logger, base_writer):
    root_dir   = cfg["dataset"]["args"]["root_dir"]
    modalities = cfg["dataset"]["args"]["modalities"]
    all_groups = discover_group_ids(root_dir, modalities)

    if is_main_process() and base_logger:
        base_logger.info(f"LOGO CV over groups: {all_groups}")

    base_log_dir  = cfg["logging"]["log_dir"]
    base_ckpt_dir = cfg["logging"]["checkpoint_dir"]

    start_idx = 0
    resume_ckpt_for_first_fold: Optional[str] = None
    if args.resume:
        start_idx, resume_ckpt_for_first_fold = find_resume_state_for_logo(base_ckpt_dir, all_groups, args.resume_path)
        if start_idx >= len(all_groups):
            if is_main_process() and base_logger:
                base_logger.info("All LOGO folds already completed. Nothing to do.")
            return

    for i in range(start_idx, len(all_groups)):
        held_out = all_groups[i]
        log_dir, ckpt_dir = fold_dirs(base_log_dir, base_ckpt_dir, held_out)

        if is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
        if distributed:
            dist.barrier()

        fold_logger, fold_writer = (setup_logger(log_dir) if is_main_process() else (None, None))
        if is_main_process() and fold_logger:
            fold_logger.info(f"====== LOGO fold {i+1}/{len(all_groups)}: held-out group {held_out} ======")
        # re-seed per fold
        base_seed = cfg["training"].get("seed", 42)
        set_seed(base_seed + i, add_rank=True)

        train_dataset, val_dataset = build_datasets(cfg, logo_held_out=held_out)
        train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset, distributed)

        model = build_model(cfg).to(device)

        if is_main_process() and fold_logger:
            log_training_hyperparams(cfg, fold_logger)
            log_model_hyperparams(cfg, fold_logger)
            log_param_counts(model, fold_logger)

        if distributed:
            if device.type != "cuda":
                raise RuntimeError("CPU distributed not supported; use CUDA or run single-process.")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        optimizer = build_optimizer(model, cfg)

        accum = max(1, cfg["training"].get("accum_steps", 1))
        total_steps = math.ceil(len(train_loader) / accum) * cfg["training"]["epochs"]
        scheduler_cfg = {
            "warmup_steps" : int(cfg["training"].get("warmup_ratio", 0) * total_steps),
            "max_steps"    : total_steps,
            "min_lr"       : cfg["training"].get("min_lr", 0.0),
            "type"         : cfg["training"].get("scheduler", ""),
        }
        scheduler = build_scheduler(optimizer, scheduler_cfg)

        trainer = Trainer(
            cfg          = cfg,
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            optimizer    = optimizer,
            scheduler    = scheduler,
            device       = device,
            logger       = fold_logger,
            writer       = fold_writer,
            world_size   = get_world_size(),
        )

        if args.resume:
            if i == start_idx and resume_ckpt_for_first_fold is not None:
                resume_path = resume_ckpt_for_first_fold
            else:
                resume_path = find_latest_checkpoint(ckpt_dir)
            if resume_path is not None:
                trainer.resume_from(resume_path)
            else:
                if is_main_process() and fold_logger:
                    fold_logger.info("No checkpoint found for this fold — starting fresh")

        # train this fold
        trainer.train(
            epochs=cfg["training"]["epochs"],
            ckpt_dir=ckpt_dir,
            validate_interval=cfg["training"].get("val_interval", 1),
            checkpoint_interval=cfg["training"].get("checkpoint_interval", 1),
        )

        mark_fold_done(ckpt_dir, {"held_out_group": held_out, "fold_index": i})

        if distributed:
            dist.barrier()

# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank, device = init_distributed_mode()

    log_dir = cfg["logging"]["log_dir"]
    ckpt_dir = cfg["logging"]["checkpoint_dir"]

    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    logger, writer = (setup_logger(log_dir) if is_main_process() else (None, None))

    set_seed(cfg["training"].get("seed", 42), add_rank=True)

    try:
        if args.logo:
            run_logo_cv(cfg, args, device, local_rank, distributed, logger, writer)
        else:
            run_training(cfg, args, device, local_rank, distributed, logger, writer)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
