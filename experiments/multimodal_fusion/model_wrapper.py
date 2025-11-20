from __future__ import annotations
from typing import Dict, List, Optional, Callable, Tuple, Any
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.SyntalNet import SyntalNet
from corruptions import (
    modality_dropout,
    modality_noise,
    modality_shuffle,
    modality_rescale,
)


class CorruptedSyntalNet(nn.Module):
    """
    Wrapper around SyntalNet that applies corruptions to branch outputs
    before multimodal fusion.
    
    The corruption is applied to the list of branch embeddings (z_branch)
    before they are passed to mm_fusion.
    """
    def __init__(
        self,
        model: SyntalNet,
        corruption_fn: Optional[Callable[[List[torch.Tensor], Optional[torch.Tensor]], 
                                        Tuple[List[torch.Tensor], Optional[torch.Tensor]]]] = None
    ):
        super().__init__()
        self.base_model = model
        self.corruption_fn = corruption_fn
        
        # Copy attributes for transparency
        self.branches = model.branches
        self.mm_fusion = model.mm_fusion
        self.individual_classifier = model.individual_classifier
        self.group_classifier = model.group_classifier
    
    def forward(self, batch_data, epoch=None):
        """
        Forward pass with corruption injection before multimodal fusion.
        
        Args:
            batch_data: Batch data dict with branch-level inputs
            epoch: Training epoch (unused in evaluation)
        
        Returns:
            (z_outs, logits): Features and classification logits
        """
        # Extract branch outputs
        z_branch = []
        for bname, branch in self.base_model.branches.items():
            x_dict = {m: batch_data[bname.lower()][m][0] for m in branch.mods}
            m_dict = {m: batch_data[bname.lower()][m][1] for m in branch.mods}
            z_b = branch(x_dict, m_dict, epoch)  # (B*P, shared_dim)
            z_branch.append(z_b)
        
        # Apply corruption if specified
        if self.corruption_fn is not None and len(self.base_model.branches) > 1:
            # Build availability mask (all available by default)
            B = z_branch[0].shape[0]
            M = len(z_branch)
            device = z_branch[0].device
            avail = torch.ones(B, M, device=device, dtype=z_branch[0].dtype)
            
            # Apply corruption
            z_branch, avail = self.corruption_fn(z_branch, avail)
        
        # Multimodal fusion
        if len(self.base_model.branches) > 1:
            z = self.base_model.mm_fusion(z_branch)
        else:
            z = z_branch[0]
        
        z_outs: Dict[str, Any] = {"backbone": z}
        logits: Dict[str, Dict[str, torch.Tensor]] = {}
        
        if self.base_model.individual_classifier:
            ind_features, ind_logits = self.base_model.individual_classifier(z, epoch)
            z_outs.update({'individual': ind_features})
            logits.update({'individual': ind_logits})
        
        if self.base_model.group_classifier:
            grp_features, grp_logits = self.base_model.group_classifier(z, epoch)
            z_outs.update({'group': grp_features})
            logits.update({'group': grp_logits})
        
        return z_outs, logits
    
    def update_prototypes(self, z_dict, y_dict):
        """Delegate to base model."""
        return self.base_model.update_prototypes(z_dict, y_dict)


def create_corruption_fn(
    corruption_type: str,
    param: float | int
) -> Optional[Callable[[List[torch.Tensor], Optional[torch.Tensor]], 
                      Tuple[List[torch.Tensor], Optional[torch.Tensor]]]]:
    """
    Create a corruption function with the specified parameter.
    
    Args:
        corruption_type: One of "modality_dropout", "modality_noise", 
                        "modality_shuffle", "modality_rescale", or None
        param: Parameter value for the corruption
    
    Returns:
        Callable that takes (z_list, avail) and returns (z_list_out, avail_out),
        or None if corruption_type is None or param indicates no corruption
    """
    # Check if this is a "no corruption" case
    no_corruption_conditions = {
        "modality_dropout": param == 0.0,
        "modality_noise": param == 0.0,
        "modality_shuffle": param == 0.0,
        "modality_rescale": param == 1.0,
    }
    
    if corruption_type is None or no_corruption_conditions.get(corruption_type, False):
        return None
    
    def corruption_wrapper(
        z_list: List[torch.Tensor],
        avail: Optional[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        if corruption_type == "modality_dropout":
            return modality_dropout(z_list, avail, float(param))
        elif corruption_type == "modality_noise":
            return modality_noise(z_list, avail, float(param))
        elif corruption_type == "modality_shuffle":
            return modality_shuffle(z_list, avail, float(param))
        elif corruption_type == "modality_rescale":
            return modality_rescale(z_list, avail, float(param))
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return corruption_wrapper

