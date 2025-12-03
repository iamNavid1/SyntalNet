"""
Fusion Variants for Reliability-Switch Experiments

Provides a clean interface to instantiate different multimodal fusion variants.
"""

import torch.nn as nn
from typing import Optional, Dict, Any

from models.multimodal_fusion import GLR_X, UniformAvgFusion, ConcatMLP


AVAILABLE_VARIANTS = {
    "glrx": GLR_X,
    "uniform_avg": UniformAvgFusion,
    "concat_mlp": ConcatMLP,
}


def build_fusion_variant(
    variant_name: str,
    num_mod: int,
    dims_mod: int,
    dim_out: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Build a multimodal fusion module by name.
    
    Args:
        variant_name: One of "glrx", "uniform_avg", "concat_mlp"
        num_mod: Number of modalities (branches)
        dims_mod: Dimension of each modality embedding
        dim_out: Output dimension (defaults to dims_mod if None)
        **kwargs: Additional arguments specific to each variant
        
    Returns:
        Fusion module instance
    """
    variant_name = variant_name.lower().strip()
    
    if variant_name not in AVAILABLE_VARIANTS:
        raise ValueError(
            f"Unknown fusion variant: {variant_name}. "
            f"Available variants: {list(AVAILABLE_VARIANTS.keys())}"
        )
    
    fusion_class = AVAILABLE_VARIANTS[variant_name]
    
    # Build the module
    fusion = fusion_class(
        num_mod=num_mod,
        dims_mod=dims_mod,
        dim_out=dim_out,
        **kwargs
    )
    
    return fusion


def get_variant_config(variant_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a fusion variant.
    
    Args:
        variant_name: One of "glrx", "uniform_avg", "concat_mlp"
        
    Returns:
        Dictionary of default hyperparameters
    """
    variant_name = variant_name.lower().strip()
    
    configs = {
        "glrx": {
            "rank_pair": 12,
            "alloc_hidden": 24,
            "eps_floor": 0.03,
            "p_drop": 0.1,
        },
        "uniform_avg": {
            "use_ln": True,
        },
        "concat_mlp": {
            "hidden": None,  # Will default to 2 * dims_mod
            "p_drop": 0.1,
        },
    }
    
    return configs.get(variant_name, {})


def variant_has_allocation_tracking(variant_name: str) -> bool:
    """Check if a variant supports allocation tracking (like GLR-X)."""
    variant_name = variant_name.lower().strip()
    return variant_name in ["glrx", "glr_x"]

