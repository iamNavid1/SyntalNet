from __future__ import annotations

import os
import math
import contextlib
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from engine.validator import Validator
from engine.utils import BuildAutocastKWargs, modalities_to_branches
from utils.losses import ClassBalancedFocalLoss, ClassBalancedCELoss


class FineTuner:
    """
    Fine-tuning trainer that freezes the encoder and only trains classifiers.
    """
    
    def __init__(
        self,
        cfg: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        logger,
        writer=None,
        world_size: int = 1,
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
        
        self.grad_clip = cfg.get("training").get("grad_clip", 0.0)
        self.accum_steps = cfg.get("training").get("accum_steps", 1)
        
        self.autocast_kwargs = BuildAutocastKWargs(cfg, device)
        self.scaler = self._build_grad_scaler()
        
        self.grp_as_ind = cfg.get("dataset").get("grp_as_ind", False)
        
        # Build loss criteria
        import yaml
        counts_cfg = cfg.get("dataset").get("cls_count_dir")
        label_type = cfg.get("dataset").get("args").get("label_type", "kernel")
        smoothing = cfg.get("training").get("label_smoothing", 0.0)
        beta = cfg.get("training").get("cb_beta", 0.999)
        gamma = cfg.get("training").get("focal_gamma", 3.0)
        loss_type = cfg.get("training").get("loss_type", "multiclass")
        tau = cfg.get("training").get("la_tau", 0.)
        
        if isinstance(counts_cfg, str):
            with open(counts_cfg, "r") as f:
                counts_cfg = yaml.safe_load(f)
        
        self.criteria = self._build_criteria(
            loss_type, counts_cfg, label_type, smoothing, beta, gamma, tau
        )
        
        self.validator = Validator(self.device, self.autocast_kwargs, self.grp_as_ind)
        
        # Freeze encoder/backbone
        self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze all parameters except classifiers."""
        model_ref = getattr(self.model, "module", self.model)
        
        # Freeze branches (encoders + fusion)
        if hasattr(model_ref, "branches"):
            for param in model_ref.branches.parameters():
                param.requires_grad = False
        
        # Freeze multimodal fusion
        if hasattr(model_ref, "mm_fusion") and model_ref.mm_fusion is not None:
            for param in model_ref.mm_fusion.parameters():
                param.requires_grad = False
        
        # Ensure classifiers are trainable
        if hasattr(model_ref, "individual_classifier") and model_ref.individual_classifier is not None:
            for param in model_ref.individual_classifier.parameters():
                param.requires_grad = True
        
        if hasattr(model_ref, "group_classifier") and model_ref.group_classifier is not None:
            for param in model_ref.group_classifier.parameters():
                param.requires_grad = True
        
        # Log parameter counts
        if self.logger:
            total_params = sum(p.numel() for p in model_ref.parameters())
            trainable_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            self.logger.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
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
        tau: float,
    ) -> Dict[str, Dict[str, nn.Module]]:
        criteria: Dict[str, Dict[str, nn.Module]] = defaultdict(dict)
        
        counts_cfg = counts_cfg.get(label_type)
        if self.grp_as_ind and "group" in counts_cfg and "individual" in counts_cfg:
            for key, values in counts_cfg["group"].items():
                counts_cfg["individual"][key] = [v * 3 for v in values]
        
        specs = []
        model_ref = getattr(self.model, "module", self.model)
        if getattr(model_ref, "individual_classifier", None) is not None:
            specs.append(("individual", model_ref.individual_classifier, counts_cfg.get("individual")))
        if getattr(model_ref, "group_classifier", None) is not None:
            specs.append(("group", model_ref.group_classifier, counts_cfg.get("group")))
        
        for split_name, classifier, counts_map in specs:
            for head_name in classifier.heads:
                c = torch.tensor(counts_map.get(head_name), dtype=torch.float, device=self.device)
                if loss_type == "ce":
                    criteria[split_name][head_name] = ClassBalancedCELoss(
                        c, beta, tau, smoothing).to(self.device)
                else:
                    criteria[split_name][head_name] = ClassBalancedFocalLoss(
                        c, beta, gamma, tau, smoothing, loss_type).to(self.device)
        
        return criteria
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        
        running_loss = 0.0
        running_count = 0
        steps = 0
        is_ddp = hasattr(self.model, "no_sync")
        
        for step, (batch_data, batch_labels) in enumerate(self.train_loader):
            steps += 1
            batch_data = {k: (v[0].to(self.device), v[1].to(self.device)) for k, v in batch_data.items()}
            batch_data = modalities_to_branches(batch_data)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}
            
            ind_label, grp_label = self._prepare_label_views(batch_labels, self.grp_as_ind)
            
            with autocast(**self.autocast_kwargs):
                features, logits = self.model(batch_data, epoch=epoch)
                loss = self._compute_loss(logits, ind_label, grp_label)
            
            # Update prototypes
            proto_targets = self._build_proto_targets(logits, ind_label, grp_label)
            with torch.no_grad():
                self.model.update_prototypes(features, proto_targets)
            
            do_sync = ((step + 1) % self.accum_steps == 0)
            ctx = self.model.no_sync() if (is_ddp and not do_sync) else contextlib.nullcontext()
            with ctx:
                self.scaler.scale(loss / self.accum_steps).backward()
            
            if do_sync:
                self._optimizer_step()
            
            batch_size = batch_labels.get("group", batch_labels.get("individual")).shape[0]
            loss_val = loss.item() * batch_size
            count_val = batch_size
            
            if torch.distributed.is_initialized():
                tensor = torch.tensor([loss_val, count_val], device=self.device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                loss_val, count_val = tensor.tolist()
            
            running_loss += loss_val
            running_count += count_val
        
        # Final optimizer step if needed
        if steps > 0 and (steps % self.accum_steps != 0):
            self._optimizer_step()
        
        avg_loss = running_loss / running_count if running_count > 0 else 0.0
        return avg_loss
    
    def validate(self) -> Tuple[Dict, float]:
        """Run validation and return metrics and loss."""
        if self.val_loader is None:
            return {}, float('inf')
        
        metrics, val_loss = self.validator(self.model, self.val_loader, self._compute_loss)
        if val_loss is not None:
            metrics["loss"] = val_loss
        return metrics, val_loss
    
    def _prepare_label_views(
        self, 
        labels: Dict[str, torch.Tensor],
        groups_as_individuals: bool = False
    ):
        ind_label = None
        grp_label = None
        
        if "individual" in labels:
            B, P, L = labels["individual"].shape
            ind_label = labels["individual"].view(B * P, L)
        else:
            B = P = L = None
        
        if "group" in labels:
            if groups_as_individuals:
                grp_label_expanded = labels["group"].repeat_interleave(P, dim=0)
                ind_label = torch.cat([ind_label, grp_label_expanded], dim=-1) \
                    if ind_label is not None else grp_label_expanded
            else:
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
        
        assert total_loss is not None, "No logits matched labels to compute loss."
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

