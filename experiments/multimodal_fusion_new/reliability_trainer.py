"""
Reliability-Switch Trainer

Trains multimodal fusion variants with selective corruption of one modality per sample.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from utils.losses import ClassBalancedFocalLoss, ClassBalancedCELoss
from utils.metrics import compute_classification_metrics_from_logits
from engine.utils import modalities_to_branches
from experiments.multimodal_fusion_new.corruptions import (
    ReliabilitySwitchConfig,
    ReliabilitySwitchCorruptor,
)
from experiments.multimodal_fusion_new.fusion_variants import variant_has_allocation_tracking


class ReliabilityTrainer:
    """Trainer for reliability-switch experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        base_config: Dict[str, Any],
        corruption_config: Optional[ReliabilitySwitchConfig] = None,
        logger: Optional[Any] = None,
        variant_name: str = "glrx",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.variant_name = variant_name
        
        # Create corruptor from config
        if corruption_config is None:
            corruption_config = ReliabilitySwitchConfig()
        self.corruptor = ReliabilitySwitchCorruptor(corruption_config)
        self.corruption_config = corruption_config
        
        # Build loss criteria (same as engine.trainer)
        self.criteria = self._build_criteria(base_config)
        self.grp_as_ind = base_config.get("dataset", {}).get("grp_as_ind", False)
        
        # Cache head names for metrics
        model_ref = getattr(self.model, "module", self.model)
        self.ind_heads = list(getattr(getattr(model_ref, "individual_classifier", None), "heads", []))
        self.grp_heads = list(getattr(getattr(model_ref, "group_classifier", None), "heads", []))
        
        # Track allocation if variant supports it
        self.track_allocation = variant_has_allocation_tracking(variant_name)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_epoch = -1
    
    def _build_criteria(self, cfg: Dict[str, Any]) -> Dict[str, Dict[str, nn.Module]]:
        """Build loss criteria for each classification head."""
        criteria: Dict[str, Dict[str, nn.Module]] = defaultdict(dict)
        
        # Load class counts
        counts_cfg = cfg.get("dataset", {}).get("cls_count_dir")
        if isinstance(counts_cfg, str):
            with open(counts_cfg, "r") as f:
                counts_cfg = yaml.safe_load(f)
        
        label_type = cfg.get("dataset", {}).get("args", {}).get("label_type", "kernel")
        counts_cfg = counts_cfg.get(label_type, {})
        
        # Get training hyperparameters
        training_cfg = cfg.get("training", {})
        smoothing = training_cfg.get("label_smoothing", 0.0)
        beta = training_cfg.get("cb_beta", 0.999)
        gamma = training_cfg.get("focal_gamma", 3.0)
        tau = training_cfg.get("la_tau", 0.0)
        loss_type = training_cfg.get("loss_type", "multiclass")
        
        # Build criteria for each classifier
        model_ref = getattr(self.model, "module", self.model)
        
        if hasattr(model_ref, "individual_classifier") and model_ref.individual_classifier is not None:
            classifier = model_ref.individual_classifier
            counts_map = counts_cfg.get("individual", {})
            for head_name in classifier.heads:
                c = torch.tensor(counts_map.get(head_name, [1] * 5), dtype=torch.float, device=self.device)
                if loss_type == "ce":
                    criteria["individual"][head_name] = ClassBalancedCELoss(
                        c, beta, tau, smoothing
                    ).to(self.device)
                else:
                    criteria["individual"][head_name] = ClassBalancedFocalLoss(
                        c, beta, gamma, tau, smoothing, loss_type
                    ).to(self.device)
        
        if hasattr(model_ref, "group_classifier") and model_ref.group_classifier is not None:
            classifier = model_ref.group_classifier
            counts_map = counts_cfg.get("group", {})
            for head_name in classifier.heads:
                c = torch.tensor(counts_map.get(head_name, [1] * 5), dtype=torch.float, device=self.device)
                if loss_type == "ce":
                    criteria["group"][head_name] = ClassBalancedCELoss(
                        c, beta, tau, smoothing
                    ).to(self.device)
                else:
                    criteria["group"][head_name] = ClassBalancedFocalLoss(
                        c, beta, gamma, tau, smoothing, loss_type
                    ).to(self.device)
        
        return criteria
    
    def _prepare_label_views(
        self,
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Flatten labels to match classifier expectations, mirroring engine.Trainer.
        Returns:
            ind_label: (B*P, L_ind) or None
            grp_label: (B, L_grp) or None (unless grp_as_ind=True)
        """
        ind_label = None
        grp_label = None

        if "individual" in labels:
            B, P, L = labels["individual"].shape
            ind_label = labels["individual"].view(B * P, L)
        else:
            B = P = L = None

        if "group" in labels:
            if self.grp_as_ind and B is not None and P is not None:
                grp_label_expanded = labels["group"].repeat_interleave(P, dim=0)
                ind_label = (
                    torch.cat([ind_label, grp_label_expanded], dim=-1)
                    if ind_label is not None else grp_label_expanded
                )
            else:
                grp_label = labels["group"]

        return ind_label, grp_label

    def _compute_loss(
        self,
        logits: Dict[str, Dict[str, torch.Tensor]],
        ind_label: Optional[torch.Tensor],
        grp_label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute total loss across all heads, mirroring engine.Trainer."""
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

        if total_loss is None:
            raise ValueError("No logits matched labels to compute loss.")

        return total_loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with reliability-switch corruption."""
        self.model.train()
        
        # Ensure branches stay in eval mode (frozen)
        if hasattr(self.model, 'branches'):
            self.model.branches.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_preds = {
            "individual": {head: [] for head in self.ind_heads},
            "group": {head: [] for head in self.grp_heads},
        }
        all_targets = {
            "individual": {head: [] for head in self.ind_heads},
            "group": {head: [] for head in self.grp_heads},
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        
        for batch_idx, (batch_modalities, batch_labels) in enumerate(pbar):
            # Move data to device and group by branches
            batch_modalities = {
                k: (v[0].to(self.device), v[1].to(self.device))
                for k, v in batch_modalities.items()
            }
            branch_data = modalities_to_branches(batch_modalities)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            # Forward through frozen branches to get z_branch embeddings
            with torch.no_grad():
                self.model.branches.eval()
                z_branch = []
                for bname, branch in self.model.branches.items():
                    key = bname.lower()
                    if key not in branch_data:
                        continue
                    mod_dict = branch_data[key]
                    x_dict = {m: mod_dict[m][0] for m in branch.mods if m in mod_dict}
                    m_dict = {m: mod_dict[m][1] for m in branch.mods if m in mod_dict}
                    if not x_dict:
                        continue
                    z_b = branch(x_dict, m_dict, epoch)
                    z_branch.append(z_b)
            
            if not z_branch:
                continue

            # Apply reliability-switch corruption
            z_branch_corrupted = self.corruptor(z_branch)
            
            # Forward through fusion and classifiers
            if len(z_branch_corrupted) > 1:
                z = self.model.mm_fusion(z_branch_corrupted)
            else:
                z = z_branch_corrupted[0]
            
            # Classification
            z_outs: Dict[str, Any] = {"backbone": z}
            logits: Dict[str, Dict[str, torch.Tensor]] = {}
            
            if self.model.individual_classifier:
                ind_features, ind_logits = self.model.individual_classifier(z, epoch)
                z_outs.update({'individual': ind_features})
                logits.update({'individual': ind_logits})
            
            if self.model.group_classifier:
                grp_features, grp_logits = self.model.group_classifier(z, epoch)
                z_outs.update({'group': grp_features})
                logits.update({'group': grp_logits})
            
            # Prepare labels and compute loss
            ind_label, grp_label = self._prepare_label_views(batch_labels)
            loss = self._compute_loss(logits, ind_label, grp_label)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update scheduler if step-based
            if hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions and targets
            for split in ['individual', 'group']:
                if split not in logits:
                    continue
                for idx, (head, head_logits) in enumerate(logits[split].items()):
                    all_preds[split][head].append(head_logits.detach().cpu())
                    if split == "individual" and ind_label is not None:
                        y = ind_label[:, idx].detach().cpu()
                    elif split == "group" and grp_label is not None:
                        y = grp_label[:, idx].detach().cpu()
                    else:
                        continue
                    all_targets[split][head].append(y)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = self._compute_epoch_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate without corruption."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_preds = {
            "individual": {head: [] for head in self.ind_heads},
            "group": {head: [] for head in self.grp_heads},
        }
        all_targets = {
            "individual": {head: [] for head in self.ind_heads},
            "group": {head: [] for head in self.grp_heads},
        }
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
        
        for batch_modalities, batch_labels in pbar:
            # Move to device and group by branches
            batch_modalities = {
                k: (v[0].to(self.device), v[1].to(self.device))
                for k, v in batch_modalities.items()
            }
            branch_data = modalities_to_branches(batch_modalities)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            # Forward through branches (no corruption)
            self.model.branches.eval()
            z_branch = []
            for bname, branch in self.model.branches.items():
                key = bname.lower()
                if key not in branch_data:
                    continue
                mod_dict = branch_data[key]
                x_dict = {m: mod_dict[m][0] for m in branch.mods if m in mod_dict}
                m_dict = {m: mod_dict[m][1] for m in branch.mods if m in mod_dict}
                if not x_dict:
                    continue
                z_b = branch(x_dict, m_dict, epoch)
                z_branch.append(z_b)

            if not z_branch:
                continue

            if len(z_branch) > 1:
                z = self.model.mm_fusion(z_branch)
            else:
                z = z_branch[0]

            # Classification
            logits: Dict[str, Dict[str, torch.Tensor]] = {}
            if self.model.individual_classifier:
                _, ind_logits = self.model.individual_classifier(z, epoch)
                logits["individual"] = ind_logits
            if self.model.group_classifier:
                _, grp_logits = self.model.group_classifier(z, epoch)
                logits["group"] = grp_logits

            # Compute loss
            ind_label, grp_label = self._prepare_label_views(batch_labels)
            loss = self._compute_loss(logits, ind_label, grp_label)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions and targets
            for split in ['individual', 'group']:
                if split not in logits:
                    continue
                for idx, (head, head_logits) in enumerate(logits[split].items()):
                    all_preds[split][head].append(head_logits.detach().cpu())
                    if split == "individual" and ind_label is not None:
                        y = ind_label[:, idx].detach().cpu()
                    elif split == "group" and grp_label is not None:
                        y = grp_label[:, idx].detach().cpu()
                    else:
                        continue
                    all_targets[split][head].append(y)
        
        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = self._compute_epoch_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str,
        validate_interval: int = 1,
        save_interval: int = 5,
    ):
        """Main training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Starting reliability-switch training for {num_epochs} epochs")
            self.logger.info(f"Corruption probability: {self.corruption_config.p_corrupt}")
            self.logger.info(f"Corruption mix: dropout={self.corruption_config.mix_dropout:.2f}, "
                           f"noise={self.corruption_config.mix_noise:.2f}, "
                           f"shuffle={self.corruption_config.mix_shuffle:.2f}")
            self.logger.info(f"Corruption params: dropout_scale={self.corruption_config.dropout_scale:.2f}, "
                           f"noise_level={self.corruption_config.noise_level:.2f}, "
                           f"shuffle_fraction={self.corruption_config.shuffle_fraction:.2f}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | "
                    f"F1: {train_metrics.get('f1_macro', 0.0):.4f}"
                )
            
            # Validate
            if epoch % validate_interval == 0:
                val_metrics = self.validate(epoch)
                
                if self.logger:
                    self.logger.info(
                        f"Epoch {epoch:02d} | Val Loss: {val_metrics['loss']:.4f} | "
                        f"F1: {val_metrics.get('f1_macro', 0.0):.4f}"
                    )
                
                # Update scheduler if epoch-based
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_epoch = epoch
                    self._save_checkpoint(checkpoint_dir, epoch, is_best=True)
                    if self.logger:
                        self.logger.info(f"New best model at epoch {epoch}")
            
            # Periodic save
            if epoch % save_interval == 0:
                self._save_checkpoint(checkpoint_dir, epoch, is_best=False)
        
        if self.logger:
            self.logger.info(f"Training complete. Best epoch: {self.best_epoch}")
    
    def _compute_epoch_metrics(
        self,
        all_preds: Dict[str, Dict[str, list]],
        all_targets: Dict[str, Dict[str, list]],
    ) -> Dict[str, float]:
        """Compute metrics across all heads and splits."""
        metrics = {}
        
        all_f1 = []
        all_auroc = []
        all_auprc = []
        
        for split, heads in (("individual", self.ind_heads), ("group", self.grp_heads)):
            for head in heads:
                if all_preds[split][head]:
                    preds = torch.cat(all_preds[split][head], dim=0)
                    targets = torch.cat(all_targets[split][head], dim=0)

                    head_metrics = compute_classification_metrics_from_logits(preds, targets)

                    # Store per-head metrics
                    prefix = f"{split}_{head}"
                    metrics[f"{prefix}_f1"] = head_metrics.get('f1_macro', 0.0)
                    metrics[f"{prefix}_auroc"] = head_metrics.get('auroc', 0.0)
                    metrics[f"{prefix}_auprc"] = head_metrics.get('auprc', 0.0)

                    all_f1.append(head_metrics.get('f1_macro', 0.0))
                    all_auroc.append(head_metrics.get('auroc', 0.0))
                    all_auprc.append(head_metrics.get('auprc', 0.0))
        
        # Average metrics
        if all_f1:
            metrics['f1_macro'] = np.mean(all_f1)
            metrics['auroc'] = np.mean(all_auroc)
            metrics['auprc'] = np.mean(all_auprc)
        
        return metrics
    
    def _move_to_device(self, batch_data: dict) -> dict:
        """Recursively move batch data to device."""
        result = {}
        for key, value in batch_data.items():
            if isinstance(value, dict):
                result[key] = self._move_to_device(value)
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    result[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for v in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        
        # Save regular checkpoint
        path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

