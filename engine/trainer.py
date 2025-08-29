from __future__ import annotations

import os
import yaml
import contextlib
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.losses import BaseLoss, ClassBalancedFocalLoss, ClassBalancedCELoss
from engine.utils import BuildAutocastKWargs, modalities_to_branches, format_metrics
from engine.validator import Validator


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        logger,
        writer: SummaryWriter,
        world_size: int = 1,
        start_epoch: int = 0,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.world_size = world_size
        self.start_epoch = start_epoch
        self.global_step = 0

        self.grad_clip = cfg.get("grad_clip", 0.0)
        self.accum_steps = cfg.get("accum_steps", 1)

        self.autocast_kwargs = BuildAutocastKWargs(cfg, device)
        self.scaler = self._build_grad_scaler()

        counts_cfg = cfg.get("dataset").get("cls_count_dir")
        label_type = cfg.get("dataset").get("args").get("label_type")
        smoothing = cfg.get("training").get("label_smoothing", 0.0)
        beta = cfg.get("training").get("cb_beta", 0.999)
        gamma = cfg.get("training").get("focal_gamma", 2.0)
        loss_type = cfg.get("training").get("loss_type", "focal")
        self.criteria = self._build_criteria(loss_type, counts_cfg, label_type, smoothing, beta, gamma)

        self.validator = Validator(self.device, self.autocast_kwargs)

    # ------------------------- public API -----------------------------

    def train(
        self,
        epochs: int,
        ckpt_dir: str,
        validate_interval: int = 1,
        checkpoint_interval: int = 1,
    ):
        os.makedirs(ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, epochs):
            if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            self._train_one_epoch(epoch)

            val_loss = None
            if self.val_loader is not None and (epoch + 1) % validate_interval == 0:
                metrics, val_loss = self.validate()
                self.logger.info(f"Validation @ epoch {epoch+1}: {format_metrics(metrics)}")

            save_best = False
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best = True

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if self._is_main() and (
                ((epoch + 1) % checkpoint_interval == 0) 
                or ((epoch + 1) == epochs)
                or save_best
            ):
                print(f"Saving checkpoint for epoch {epoch+1} to {ckpt_dir}...")
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
                self.save_checkpoint(ckpt_path, epoch)

        self.close()


    def validate(self):
        if self.val_loader is None:
            return {}, None
        results, val_loss = self.validator(self.model, self.val_loader, self._compute_loss)
        if val_loss is not None:
            results["loss"] = val_loss
        if self._is_main():
            if val_loss is not None:
                self.writer.add_scalar("val/loss", val_loss, self.global_step)
            for group_name, heads in results.items():
                if group_name == "loss":
                    continue
                for head_name, res in heads.items():
                    for k, v in res.items():
                        tag = f"val/{group_name}_{head_name}_{k}"
                        if k == "confusion_matrix":
                            cm = torch.as_tensor(v, dtype=torch.float32)
                            cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
                            cm = torch.nn.functional.interpolate(cm[None, None], size=(256, 256), mode="nearest")[0, 0]
                            self.writer.add_image(tag, cm, self.global_step, dataformats="HW")
                        elif isinstance(v, (list, tuple, np.ndarray)) and np.size(v) > 1:
                            vals = v if isinstance(v, (list, tuple)) else v.tolist()
                            for i, vi in enumerate(vals):
                                self.writer.add_scalar(f"{tag}/class_{i+1}", float(vi), self.global_step)
                        else:
                            self.writer.add_scalar(tag, float(np.array(v).squeeze()), self.global_step)

        return results, val_loss


    def save_checkpoint(self, path: str, epoch: int):
        if not self._is_main():
            return
        state = {
            "model": getattr(self.model, "module", self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": self.global_step,
        }
        torch.save(state, path)


    def resume_from(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        getattr(self.model, "module", self.model).load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.global_step = state.get("global_step", 0)
        self.start_epoch = state.get("epoch")
        self.logger.info(f"Resumed from checkpoint {path} at epoch {state.get('epoch')}")
        return state

    # ------------------------- private API -----------------------------

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        steps = 0
        is_ddp = hasattr(self.model, "no_sync")

        for step, (batch_data, batch_labels) in enumerate(self.train_loader):
            steps += 1
            batch_data = {k: (v[0].to(self.device), v[1].to(self.device)) for k, v in batch_data.items()}
            batch_data = modalities_to_branches(batch_data)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            ind_label, grp_label = self._prepare_label_views(batch_labels)

            with autocast(**self.autocast_kwargs):
                features, logits = self.model(batch_data, epoch=epoch)
                loss = self._compute_loss(logits, ind_label, grp_label)

            proto_targets = self._build_proto_targets(logits, ind_label, grp_label)

            with torch.no_grad():
                self.model.update_prototypes(features, proto_targets)

            do_sync = ((step + 1) % self.accum_steps == 0)
            ctx = self.model.no_sync() if (is_ddp and not do_sync) else contextlib.nullcontext()
            with ctx:
                self.scaler.scale(loss / self.accum_steps).backward()
            if do_sync:
                self._optimizer_step()

            running_loss += loss.item()
            if self._is_main() and (step + 1) % 10 == 0:
                avg_loss = running_loss / (step + 1)
                self.logger.info(f"Epoch {epoch+1} Step {step+1}/{len(self.train_loader)} Loss {avg_loss:.4f}")
                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                for i, pg in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f"train/lr_group{i}", pg.get("lr", 0.0), self.global_step)
            self.global_step += 1

        if steps > 0 and (steps % self.accum_steps != 0):  # tail microbatch
            self._optimizer_step()


    def _prepare_label_views(self, labels: Dict[str, torch.Tensor]):
        ind_label = None
        grp_label = None
        if "individual" in labels:
            B, P, L = labels["individual"].shape
            ind_label = labels["individual"].view(B * P, L)
        if "group" in labels:
            grp_label = labels["group"]
        return ind_label, grp_label


    def _compute_loss(
        self,
        logits: Dict[str, Dict[str, torch.Tensor]],
        ind_label: Optional[torch.Tensor],
        grp_label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = None

        if "individual" in logits and ind_label is not None:
            for idx, (name, lg) in enumerate(logits["individual"].items()):
                y = ind_label[:, idx]
                loss = self.criteria["individual"][name](lg, y)
                total_loss = loss if total_loss is None else total_loss + loss

        if "group" in logits and grp_label is not None:
            for idx, (name, lg) in enumerate(logits["group"].items()):
                y = grp_label[:, idx]
                loss = self.criteria["group"][name](lg, y)
                total_loss = loss if total_loss is None else total_loss + loss

        return total_loss


    def _build_proto_targets(
        self,
        logits: Dict[str, Dict[str, torch.Tensor]],
        ind_label: Optional[torch.Tensor],
        grp_label: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        proto_targets: Dict[str, Dict[str, torch.Tensor]] = {"individual": {}, "group": {}}

        if "individual" in logits and ind_label is not None:
            for idx, (name, _lg) in enumerate(logits["individual"].items()):
                proto_targets["individual"][name] = ind_label[:, idx]

        if "group" in logits and grp_label is not None:
            for idx, (name, _lg) in enumerate(logits["group"].items()):
                proto_targets["group"][name] = grp_label[:, idx]

        return proto_targets


    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()


    def _build_grad_scaler(self) -> GradScaler:
        enabled = bool(
            self.autocast_kwargs.get("enabled", False)
            and self.device.type == "cuda"
            and self.autocast_kwargs.get("dtype") is torch.float16
        )
        return GradScaler(enabled=enabled)


    def _build_criteria(
        self,
        loss_type: str,
        counts_cfg: Dict,
        label_type: str,
        smoothing: float,
        beta: float,
        gamma: float,
    ) -> Dict[str, Dict[str, BaseLoss]]:
        criteria: Dict[str, Dict[str, BaseLoss]] = defaultdict(dict)

        if isinstance(counts_cfg, str):
            with open(counts_cfg, "r") as f:
                counts_cfg = yaml.safe_load(f)
        counts_cfg = counts_cfg.get(label_type)

        specs = []
        if getattr(self.model, "individual_classifier", None) is not None:
            specs.append(("individual", self.model.individual_classifier, counts_cfg.get("individual")))
        if getattr(self.model, "group_classifier", None) is not None:
            specs.append(("group", self.model.group_classifier, counts_cfg.get("group")))

        for split_name, classifier, counts_map in specs:
            for head_name in classifier.heads:
                c = torch.tensor(counts_map.get(head_name), dtype=torch.float, device=self.device)
                if loss_type == "ce":
                    criteria[split_name][head_name] = ClassBalancedCELoss(
                        c, beta, smoothing).to(self.device)
                else:
                    criteria[split_name][head_name] = ClassBalancedFocalLoss(
                        c, beta, gamma, smoothing).to(self.device)

        return criteria


    @staticmethod
    def _is_main() -> bool:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


    def close(self):
        if self.writer is not None:
            self.writer.close()


