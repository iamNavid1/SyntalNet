from __future__ import annotations

from typing import Callable, Dict, List, Optional
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader

from engine.utils import modalities_to_branches
from experiments.multimodal_fusion.corruptions import CORRUPTION_FUNCS
from utils.metrics import compute_classification_metrics_from_logits


CorruptionFn = Callable[[List[torch.Tensor]], List[torch.Tensor]]


def _make_corruption_fn(
    corruption_type: str, param: float, modality_idx: Optional[int]
) -> CorruptionFn:
    def apply(z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if corruption_type not in CORRUPTION_FUNCS:
            raise KeyError(f"Unknown corruption type '{corruption_type}'")

        # Per-modality corruption: simple wrapper around CORRUPTION_FUNCS
        if modality_idx is not None:
            func = CORRUPTION_FUNCS[corruption_type]
            return func(z_list, param, modality_idx)

        # "All" case: for each sample, randomly select exactly one modality to corrupt, others remain clean
        M = len(z_list)
        if M == 0 or param == 0.0:
            return z_list

        B = z_list[0].shape[0]
        device = z_list[0].device
        dtype = z_list[0].dtype

        # Random modality choice per sample: (B,)
        mod_choices = torch.randint(low=0, high=M, size=(B,), device=device)

        out = [z.clone() for z in z_list]

        for j in range(M):
            mask = (mod_choices == j)
            if not mask.any():
                continue
            z = out[j]

            if corruption_type == "dropout":
                # Zero out entire embedding for masked samples with prob=param
                keep = (torch.rand_like(mask, dtype=dtype) > param)
                effective_mask = mask & ~keep.bool()
                if effective_mask.any():
                    z[effective_mask] = 0.0

            elif corruption_type == "noise":
                sub = z[mask]
                if sub.numel() == 0:
                    continue
                mean = sub.mean(dim=1, keepdim=True)
                var = ((sub - mean) ** 2).mean(dim=1, keepdim=True)
                std = var.clamp_min(1e-6).sqrt()
                eps = torch.randn_like(sub, device=device, dtype=dtype)
                z[mask] = sub + eps * std * float(param)

            elif corruption_type == "shuffle":
                # Shuffle only within the masked subset
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() <= 1:
                    continue
                perm = idx[torch.randperm(idx.numel(), device=device)]
                z[idx] = z[perm]

            elif corruption_type == "rescale":
                z[mask] = z[mask] * float(param)

            out[j] = z

        return out

    return apply


class ReliabilityStressTester:
    """Runs clean evaluation and corruption sweeps for a trained model."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        variant_name: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.loader = loader
        self.device = device
        self.variant_name = variant_name
        self.branch_names = list(self.model.branches.keys())
        self.logger = logger

    @torch.no_grad()
    def evaluate_clean(self) -> Dict[str, Dict[str, float]]:
        preds, targets = self._run_batches(None)
        return self._compute_metrics(preds, targets)

    @torch.no_grad()
    def run_stress_tests(
        self,
        corruption_grid: Dict[str, List[float]],
        per_modality: bool,
        baseline: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for corr_type, values in corruption_grid.items():
            if self.logger:
                self.logger.info(
                    "Stress tests: uniform %s sweep over %s", corr_type, values
                )
            frames.append(
                self._sweep_corruption(corr_type, values, modality_idx=None, baseline=baseline)
            )
            if per_modality:
                for mod_idx, _ in enumerate(self.branch_names):
                    if self.logger:
                        self.logger.info(
                            "Stress tests: %s on modality=%s sweep over %s",
                            corr_type,
                            self.branch_names[mod_idx],
                            values,
                        )
                    frames.append(
                        self._sweep_corruption(corr_type, values, mod_idx, baseline=baseline)
                    )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _sweep_corruption(
        self,
        corr_type: str,
        values: List[float],
        modality_idx: Optional[int],
        baseline: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        records = []
        for val in values:
            # Treat "no corruption" as baseline when available to avoid recomputation
            if baseline is not None and abs(float(val)) < 1e-8:
                metrics = baseline
                mod_name = (
                    self.branch_names[modality_idx]
                    if modality_idx is not None
                    else "random_single"
                )
                if self.logger:
                    self.logger.info(
                        "Reusing clean baseline for %s param=%.3f (modality=%s)",
                        corr_type,
                        val,
                        self.branch_names[modality_idx] if modality_idx is not None else "random_single",
                    )
            else:
                preds, targets = self._run_batches(
                    _make_corruption_fn(corr_type, val, modality_idx)
                )
                metrics = self._compute_metrics(preds, targets)
                mod_name = (
                    self.branch_names[modality_idx]
                    if modality_idx is not None
                    else "random_single"
                )

            for split, heads in metrics.items():
                for head, head_metrics in heads.items():
                    records.append(
                        {
                            "corruption_type": corr_type,
                            "corruption_param": val,
                            "modality": mod_name,
                            "split": split,
                            "head": head,
                            "f1_macro": head_metrics.get("f1_macro", 0.0),
                            "auroc": head_metrics.get("auroc", 0.0),
                            "auprc": head_metrics.get("auprc", 0.0),
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _run_batches(
        self,
        corruption: Optional[CorruptionFn],
    ):
        preds = {"individual": {}, "group": {}}
        targets = {"individual": {}, "group": {}}
        model_ref = self.model
        model_ref.eval()

        for batch_modalities, batch_labels in self.loader:
            batch_modalities = {
                k: (v[0].to(self.device), v[1].to(self.device))
                for k, v in batch_modalities.items()
            }
            branch_data = modalities_to_branches(batch_modalities)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            z_branch: List[torch.Tensor] = []
            for bname, branch in model_ref.branches.items():
                key = bname.lower()
                mod_dict = branch_data[key]
                x_dict = {m: mod_dict[m][0] for m in branch.mods}
                m_dict = {m: mod_dict[m][1] for m in branch.mods}
                branch.eval()
                with torch.no_grad():
                    z_b = branch(x_dict, m_dict, epoch=None)
                z_branch.append(z_b)

            if corruption is not None:
                z_branch = corruption(z_branch)

            if len(z_branch) > 1:
                z = model_ref.mm_fusion(z_branch)
            else:
                z = z_branch[0]

            logits = {}
            if model_ref.individual_classifier:
                _, ind_logits = model_ref.individual_classifier(z, epoch=None)
                logits["individual"] = ind_logits
            if model_ref.group_classifier:
                _, grp_logits = model_ref.group_classifier(z, epoch=None)
                logits["group"] = grp_logits

            if "individual" in batch_labels:
                ind_labels = batch_labels["individual"].view(
                    batch_labels["individual"].shape[0]
                    * batch_labels["individual"].shape[1],
                    -1,
                )
            else:
                ind_labels = None
            grp_labels = batch_labels.get("group")

            for split in ["individual", "group"]:
                if split not in logits:
                    continue
                if split not in preds:
                    preds[split] = {}
                    targets[split] = {}
                head_labels = ind_labels if split == "individual" else grp_labels
                if head_labels is None:
                    continue
                for idx, (head, head_logits) in enumerate(logits[split].items()):
                    preds[split].setdefault(head, []).append(head_logits.detach().cpu())
                    targets[split].setdefault(head, []).append(
                        head_labels[:, idx].detach().cpu()
                    )

        return preds, targets

    @staticmethod
    def _compute_metrics(preds, targets):
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split in ["individual", "group"]:
            if split not in preds:
                continue
            summary.setdefault(split, {})
            for head, head_preds in preds[split].items():
                if not head_preds:
                    continue
                pred_tensor = torch.cat(head_preds, dim=0)
                target_tensor = torch.cat(targets[split][head], dim=0)
                metrics = compute_classification_metrics_from_logits(
                    pred_tensor, target_tensor
                )
                summary[split][head] = {
                    "accuracy": metrics.get("accuracy", 0.0),
                    "f1_macro": metrics.get("f1_macro", 0.0),
                    "auroc": metrics.get("auroc_macro", 0.0),
                    "auprc": metrics.get("auprc_macro", 0.0),
                }
        return summary

