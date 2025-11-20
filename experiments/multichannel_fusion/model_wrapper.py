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
from experiments.multichannel_fusion.corruptions import (
    stream_dropout,
    channel_dropout,
    temporal_band_mask,
    misalignment_jitter,
    energy_imbalance,
    feature_noise,
)


def apply_corruption_to_branch(
    branch: nn.Module,
    corruption_fn: Optional[Callable]
) -> nn.Module:
    """
    Apply corruption to a Branch by patching its forward method.
    This creates a wrapper that intercepts Zs and Ms before mc_fusion.
    """
    if corruption_fn is None:
        return branch
    
    # Create a wrapper module
    class CorruptedBranch(nn.Module):
        def __init__(self, original_branch, corr_fn):
            super().__init__()
            self.branch = original_branch
            self.corruption_fn = corr_fn
            # Copy essential attributes
            self.mods = original_branch.mods
            self.mods_order = original_branch.mods_order
            self.encoders = original_branch.encoders
            self.mc_fusion = original_branch.mc_fusion
            if hasattr(original_branch, 'upsamplers'):
                self.upsamplers = original_branch.upsamplers
            if hasattr(original_branch, 'single_pool'):
                self.single_pool = original_branch.single_pool
        
        def forward(self, x_dict, m_dict, epoch=None):
            # Run the branch forward but intercept before mc_fusion
            any_x = next(iter(x_dict.values()))
            B, P = any_x.shape[:2]
            
            Zs, Ms = [], []
            mods_order = self.branch.mods_order
            
            for m in mods_order:
                x = x_dict[m]
                mk = m_dict[m]
                
                # Handle upsampling
                if m in getattr(self.branch, 'upsamplers', {}):
                    N = B * P
                    xN = x.permute(0, 1, 3, 2).reshape(N, x.shape[-1], x.shape[-2])
                    xN = self.branch.upsamplers[m](xN)
                    x = xN.permute(0, 2, 1).reshape(B, P, xN.shape[-1], xN.shape[-2])
                    
                    mN = mk.permute(0, 1, 3, 2).reshape(N, mk.shape[-1], mk.shape[-2]).to(xN.dtype)
                    mN = torch.nn.functional.interpolate(mN, size=xN.shape[-1], mode='nearest')
                    mk = mN.permute(0, 2, 1).reshape(B, P, mN.shape[-1], mN.shape[-2])
                
                # Encoder
                x = x.permute(0, 1, 3, 2).unsqueeze(-1)
                mk = mk.permute(0, 1, 3, 2).unsqueeze(-1)
                ze, me = self.branch.encoders[m](x, mk)
                
                # Flatten
                ze = ze.view(B*P, ze.shape[-3], ze.shape[-2], ze.shape[-1])
                me = me.view(B*P, me.shape[-3], me.shape[-2], me.shape[-1])
                me = me.expand(-1, ze.shape[1], -1, -1).to(dtype=ze.dtype)
                
                Zs.append(ze)
                Ms.append(me)
            
            # Apply corruption
            Zs, Ms = self.corruption_fn(Zs, Ms)
            
            # Fusion
            if self.branch.mc_fusion is not None:
                z_branch = self.branch.mc_fusion(Zs, Ms)
            else:
                Z, M = Zs[0], Ms[0]
                mean = (Z*M).sum(dim=(2,3)) / M.sum(dim=(2,3)).clamp_min(1e-6)
                gem = self.branch.single_pool["gem"](Z, M)
                vec = torch.cat([mean, gem], dim=-1)
                z_branch = self.branch.single_pool["head"](vec)
            
            return z_branch
    
    return CorruptedBranch(branch, corruption_fn)


class CorruptedSyntalNet(nn.Module):
    """
    Wrapper around SyntalNet that applies corruptions to branch outputs.
    """
    def __init__(self, model: SyntalNet, corruption_fn: Optional[Callable] = None):
        super().__init__()
        self.base_model = model
        self.corruption_fn = corruption_fn
        
        # Wrap branches with corruption
        if corruption_fn is not None:
            self.branches = nn.ModuleDict()
            for name, branch in model.branches.items():
                self.branches[name] = apply_corruption_to_branch(branch, corruption_fn)
        else:
            self.branches = model.branches
        
        # Copy other attributes
        self.mm_fusion = model.mm_fusion
        self.individual_classifier = model.individual_classifier
        self.group_classifier = model.group_classifier
        
    def forward(self, batch_data, epoch=None):
        """Forward pass with corruption injection."""
        z_branch = []
        for bname, branch in self.branches.items():
            # Access mods from the branch (works for both wrapped and unwrapped)
            mods = branch.mods if hasattr(branch, 'mods') else branch.branch.mods
            x_dict = {m: batch_data[bname.lower()][m][0] for m in mods}
            m_dict = {m: batch_data[bname.lower()][m][1] for m in mods}
            z_b = branch(x_dict, m_dict, epoch)
            z_branch.append(z_b)
        
        if len(self.branches) > 1:
            z = self.mm_fusion(z_branch)
        else:
            z = z_branch[0]
        
        z_outs: Dict[str, Any] = {"backbone": z}
        logits: Dict[str, Dict[str, torch.Tensor]] = {}
        
        if self.individual_classifier:
            ind_features, ind_logits = self.individual_classifier(z, epoch)
            z_outs.update({'individual': ind_features})
            logits.update({'individual': ind_logits})
            
        if self.group_classifier:
            grp_features, grp_logits = self.group_classifier(z, epoch)
            z_outs.update({'group': grp_features})
            logits.update({'group': grp_logits})
        
        return z_outs, logits
    
    def update_prototypes(self, z_dict, y_dict):
        """Delegate to base model."""
        return self.base_model.update_prototypes(z_dict, y_dict)


def create_corruption_fn(corruption_type: str, param: float | int) -> Optional[Callable]:
    """
    Create a corruption function with the specified parameter.
    
    Args:
        corruption_type: One of "stream_dropout", "channel_dropout", 
                        "temporal_band", "jitter", "energy", "noise", or None
        param: Parameter value for the corruption
    
    Returns:
        Callable that takes (zs, ms) and returns (zs_out, ms_out),
        or None if corruption_type is None or param indicates no corruption
    """
    # Check if this is a "no corruption" case
    no_corruption_conditions = {
        "stream_dropout": param == 0.0,
        "channel_dropout": param == 0.0,
        "temporal_band": param == 0.0,
        "jitter": param == 0,
        "energy": param == 1.0,
        "noise": param == 0.0,
    }
    
    if corruption_type is None or no_corruption_conditions.get(corruption_type, False):
        return None
    
    def corruption_wrapper(zs: List[torch.Tensor], ms: List[torch.Tensor]):
        if corruption_type == "stream_dropout":
            return stream_dropout(zs, ms, float(param))
        elif corruption_type == "channel_dropout":
            return channel_dropout(zs, ms, float(param))
        elif corruption_type == "temporal_band":
            return temporal_band_mask(zs, ms, float(param))
        elif corruption_type == "jitter":
            return misalignment_jitter(zs, ms, int(param))
        elif corruption_type == "energy":
            return energy_imbalance(zs, ms, float(param))
        elif corruption_type == "noise":
            return feature_noise(zs, ms, float(param))
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return corruption_wrapper

