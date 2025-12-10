from __future__ import annotations

import copy
import json
import math
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

import models.builders as build
from experiments.ood_adaptation.finetuner import FineTuner
from experiments.multimodal_fusion.config import ExperimentConfig
from experiments.multimodal_fusion.model_utils import prepare_model
from experiments.multimodal_fusion.stress import ReliabilityStressTester
from experiments.multimodal_fusion.allocation_tracker import run_allocation_tracking
from train import (
    build_datasets,
    build_loaders,
    discover_group_ids,
    set_seed,
)
from utils.logger import setup_logger
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler


@dataclass
class FoldResources:
    fold_idx: int
    train_loader: DataLoader
    val_loader: DataLoader
    ckpt_path: str
    fold_dir: str
    logger: logging.Logger


class ReliabilityExperimentRunner:
    """Coordinates reliability-switch training + evaluation for a set of folds."""

    def __init__(self, cfg: ExperimentConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        # Load base SyntalNet config and set global seed for reproducibility
        self.base_cfg = copy.deepcopy(
            build.config(cfg.base_config_path)
        )  # type: ignore[attr-defined]
        self.base_seed = int(self.base_cfg.get("training", {}).get("seed", 42))
        # Single global seeding anchor, mirroring train.py behaviour
        set_seed(self.base_seed, add_rank=False)
        self._logo_groups = None

    def train_fold(self, fold_idx: int, run_evaluation: bool = True) -> None:
        resources = self._prepare_fold(fold_idx)
        resources.logger.info(
            "==== Fold %02d | variant=%s | phase=train ====",
            resources.fold_idx,
            self.cfg.fusion.name,
        )
        model, fusion_wrapper = prepare_model(
            self.cfg, copy.deepcopy(self.base_cfg), resources.ckpt_path, self.device
        )
        finetune_cfg = self._make_finetune_cfg()
        optimizer = build_optimizer(model, finetune_cfg)
        scheduler = self._make_scheduler(
            optimizer, finetune_cfg, len(resources.train_loader)
        )

        finetuner = FineTuner(
            cfg=finetune_cfg,
            model=model,
            train_loader=resources.train_loader,
            val_loader=resources.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            logger=resources.logger,
            writer=None,
            freeze_backbone=True,
        )

        # Re-enable multimodal fusion parameters so we train (mm_fusion + heads)
        # while keeping encoders and branch mixers frozen.
        model_ref = getattr(model, "module", model)
        if getattr(model_ref, "mm_fusion", None) is not None:
            n_total = sum(p.numel() for p in model_ref.parameters())
            for p in model_ref.mm_fusion.parameters():
                p.requires_grad = True
            n_train = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
            frozen = n_total - n_train
            resources.logger.info(
                "Reliability freeze: trainable=%d (%.2f%%), frozen=%d (%.2f%%)",
                n_train,
                100.0 * n_train / max(1, n_total),
                frozen,
                100.0 * frozen / max(1, n_total),
            )

        history = []
        best_loss = float("inf")
        best_path = os.path.join(resources.fold_dir, "checkpoints", "best.pth")
        last_path = os.path.join(resources.fold_dir, "checkpoints", "last.pth")

        resources.logger.info(
            "Training for %d epochs (val_interval=%d)",
            self.cfg.training.num_epochs,
            self.cfg.training.val_interval,
        )
        if self.cfg.training.val_interval <= 0:
            resources.logger.info(
                "Validation during training is disabled (val_interval <= 0)"
            )

        for epoch in range(self.cfg.training.num_epochs):
            train_loss = finetuner.train_epoch(epoch)

            metrics = {}
            val_loss = None
            # val_interval <= 0 means: no validation during training
            if self.cfg.training.val_interval > 0:
                do_validate = (
                    (epoch + 1) % self.cfg.training.val_interval == 0
                    or (epoch + 1) == self.cfg.training.num_epochs
                )
            else:
                do_validate = False
            if do_validate:
                metrics, val_loss = finetuner.validate()
                finetuner.step_scheduler_on_epoch(val_loss)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            if val_loss is not None:
                resources.logger.info(
                    "Epoch %02d | train=%.4f val=%.4f",
                    epoch + 1,
                    train_loss,
                    val_loss,
                )
            else:
                resources.logger.info(
                    "Epoch %02d | train=%.4f (no validation this epoch)",
                    epoch + 1,
                    train_loss,
                )

            state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }
            torch.save(state, last_path)

            if (val_loss is not None) and (val_loss < best_loss):
                best_loss = val_loss
                torch.save(state, best_path)

        fusion_wrapper.enable_corruption(False)
        if run_evaluation:
            resources.logger.info(
                "Finished training for fold %02d; evaluating last checkpoint",
                resources.fold_idx,
            )
            self._evaluate_fold(resources, last_path, history)
        else:
            resources.logger.info(
                "Finished training for fold %02d; skipping evaluation (train-only mode)",
                resources.fold_idx,
            )

    def evaluate_fold(self, fold_idx: int) -> None:
        resources = self._prepare_fold(fold_idx, create_logger=False)
        resources.logger.info(
            "==== Fold %02d | variant=%s | phase=eval-only ====",
            resources.fold_idx,
            self.cfg.fusion.name,
        )
        last_path = os.path.join(resources.fold_dir, "checkpoints", "last.pth")
        best_path = os.path.join(resources.fold_dir, "checkpoints", "best.pth")
        if os.path.isfile(last_path):
            ckpt_path = last_path
        elif os.path.isfile(best_path):
            ckpt_path = best_path
        else:
            raise FileNotFoundError(
                f"No trained checkpoint found for fold {fold_idx} "
                f"(expected {last_path} or {best_path}). Run training first."
            )
        resources.logger.info("Evaluating checkpoint: %s", ckpt_path)
        self._evaluate_fold(resources, ckpt_path, history=None)

    # ------------------------------------------------------------------ helpers
    def _prepare_fold(self, fold_idx: int, create_logger: bool = True) -> FoldResources:
        train_loader, val_loader = self._build_loaders(fold_idx)
        fold_dir = self.cfg.fold_output_dir(self.cfg.fusion.name, fold_idx)
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "results"), exist_ok=True)

        logger, writer = setup_logger(
            os.path.join(fold_dir, "logs"), name=f"reliability_fold_{fold_idx:02d}"
        )
        if writer is not None:
            writer.close()

        ckpt_path = self.cfg.checkpoint_for_fold(fold_idx)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Base checkpoint for fold {fold_idx} not found at {ckpt_path}"
            )

        logger.info(
            "Preparing fold %02d for variant=%s", fold_idx, self.cfg.fusion.name
        )
        logger.info("Base checkpoint: %s", ckpt_path)

        return FoldResources(fold_idx, train_loader, val_loader, ckpt_path, fold_dir, logger)

    def _evaluate_fold(
        self,
        resources: FoldResources,
        checkpoint_path: str,
        history: Optional[list],
    ) -> None:
        # Reseed so evaluation (including stress-test corruptions) is reproducible
        set_seed(self.base_seed + resources.fold_idx, add_rank=False)
        resources.logger.info(
            "Starting evaluation for fold %02d (clean metrics + stress tests)",
            resources.fold_idx,
        )
        model, fusion_wrapper = prepare_model(
            self.cfg, copy.deepcopy(self.base_cfg), resources.ckpt_path, self.device
        )
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state["model"])
        fusion_wrapper.enable_corruption(False)

        tester = ReliabilityStressTester(
            model=model,
            loader=resources.val_loader,
            device=self.device,
            variant_name=self.cfg.fusion.name,
            logger=resources.logger,
        )

        clean_metrics = tester.evaluate_clean()
        stress_df = tester.run_stress_tests(
            self.cfg.stress.corruption_grid,
            self.cfg.stress.per_modality,
            baseline=clean_metrics,
        )
        results_dir = os.path.join(resources.fold_dir, "results")

        clean_path = os.path.join(results_dir, "clean_metrics.csv")
        stress_path = os.path.join(results_dir, "stress_tests.csv")
        self._write_clean_metrics(clean_metrics, clean_path)
        stress_df.to_csv(stress_path, index=False)
        resources.logger.info("Saved clean metrics to %s", clean_path)
        resources.logger.info("Saved stress-test results to %s", stress_path)

        if history is not None:
            with open(
                os.path.join(results_dir, "training_history.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(history, f, indent=2)

        if self.cfg.stress.run_allocation:
            resources.logger.info("Running allocation tracking for GLR-X under noise sweep")
            alloc_path = run_allocation_tracking(
                model=model,
                dataloader=resources.val_loader,
                device=self.device,
                cfg=self.base_cfg,
                variant_name=self.cfg.fusion.name,
                fold_idx=resources.fold_idx,
                output_dir=results_dir,
                noise_levels=self.cfg.stress.allocation_noise_levels,
            )
            resources.logger.info("Saved allocation tracking CSV to %s", alloc_path)

        resources.logger.info("Completed evaluation for fold %02d", resources.fold_idx)

    def _build_loaders(self, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
        # Fold-specific seeding so k-fold splits and corruption patterns are stable
        set_seed(self.base_seed + fold_idx, add_rank=False)
        if self.cfg.cv_mode.lower() == "kfold":
            train_dataset, val_dataset = build_datasets(
                self.base_cfg, kfold_fold_idx=fold_idx
            )
        elif self.cfg.cv_mode.lower() == "logo":
            held_out = self._get_logo_groups()[fold_idx]
            train_dataset, val_dataset = build_datasets(
                self.base_cfg, logo_held_out=held_out
            )
        else:
            raise ValueError(f"Unsupported cv_mode: {self.cfg.cv_mode}")
        distributed = False
        train_loader, val_loader = build_loaders(
            self.base_cfg, train_dataset, val_dataset, distributed
        )
        return train_loader, val_loader

    def _make_finetune_cfg(self) -> Dict:
        cfg = copy.deepcopy(self.base_cfg)
        train_cfg = cfg.setdefault("training", {})
        overrides = self.cfg.training
        train_cfg["epochs"] = overrides.num_epochs
        if overrides.learning_rate is not None:
            train_cfg["learning_rate"] = overrides.learning_rate
        if overrides.weight_decay is not None:
            train_cfg["weight_decay"] = overrides.weight_decay
        if overrides.grad_clip is not None:
            train_cfg["grad_clip"] = overrides.grad_clip
        if overrides.accum_steps is not None:
            train_cfg["accum_steps"] = overrides.accum_steps
        return cfg

    def _make_scheduler(
        self, optimizer, cfg: Dict, train_len: int
    ):
        overrides = self.cfg.training
        if overrides.scheduler == "reduce_on_plateau":
            sched_cfg = {
                "type": "reduce_on_plateau",
                "factor": overrides.scheduler_factor,
                "patience": overrides.scheduler_patience,
                "threshold": overrides.scheduler_threshold,
                "min_lr": overrides.scheduler_min_lr,
                "mode": "min",
            }
            return build_scheduler(optimizer, sched_cfg)

        accum = max(1, cfg["training"].get("accum_steps", 1))
        total_steps = math.ceil(train_len / accum) * overrides.num_epochs
        sched_cfg = {
            "type": overrides.scheduler,
            "warmup_steps": int(cfg["training"].get("warmup_ratio", 0) * total_steps),
            "max_steps": total_steps,
            "min_lr": cfg["training"].get("min_lr", 0.0),
        }
        return build_scheduler(optimizer, sched_cfg)

    def _get_logo_groups(self):
        if self._logo_groups is None:
            args = self.base_cfg["dataset"]["args"]
            self._logo_groups = discover_group_ids(args["root_dir"], args["modalities"])
        return self._logo_groups

    @staticmethod
    def _write_clean_metrics(metrics, path: str) -> None:
        rows = []
        for split, heads in metrics.items():
            for head, vals in heads.items():
                row = {"split": split, "head": head}
                row.update(vals)
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

