from __future__ import annotations

import yaml
import importlib
from typing import Any, Dict, List
import torch.nn as nn

from models.multichannel_fusion import (
    BSC_X,
    BSXProjOnly,
    ConcatProjFusion,
    UniformAvgFusion
)
from models.multimodal_fusion import (
    GLR_X, 
    UniformAvgFusion,
    GatedSumOnly,
    PairwiseOnly,
    ConcatMLP
)


def model(config: Dict[str, Any]):
    cfg = config["model"]
    name = cfg["name"]
    args = cfg.get("args", {})

    module = importlib.import_module(f"models.{name}")
    model_cls = getattr(module, name)
    return model_cls(**args)


def config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def multichannel_fusion(
    variant: str,
    Cin_list: List[int],
    C: int = 128,
    out_dim: int = 128,
    **kwargs
) -> nn.Module:
    variant = variant.lower()
    if variant in ("bscx", "bscx_full"):
        return BSC_X(Cin_list, C=C, out_dim=out_dim, **kwargs)
    if variant in ("proj_only", "bscx_proj_only"):
        return BSXProjOnly(Cin_list, C=C, out_dim=out_dim)
    if variant in ("concat_proj", "bscx_concat_proj"):
        return ConcatProjFusion(Cin_list, out_dim=out_dim)
    if variant in ("uniform_avg", "bscx_uniform_avg"):
        return UniformAvgFusion(Cin_list, out_dim=out_dim)

    raise ValueError(
        f"Unknown BSCX variant '{variant}'"
        f"Valid options: 'bscx', 'bscx_full', 'proj_only', 'bscx_proj_only', 'concat_proj', 'bscx_concat_proj', 'uniform_avg', 'bscx_uniform_avg'"
    )


def multimodal_fusion(
    variant: str,
    num_mod: int,
    dims_mod: int,
    dim_out: int | None = None,
    **kwargs
) -> nn.Module:
    variant = variant.lower()
    if variant in ("glr_x", "glrx", "glr"):
        return GLR_X(num_mod=num_mod, dims_mod=dims_mod, dim_out=dim_out, **kwargs)
    if variant in ("uniform_avg", "uniform", "mean"):
        return UniformAvgFusion(num_mod=num_mod, dims_mod=dims_mod, dim_out=dim_out, **kwargs)
    if variant in ("gated_sum_only", "gated_sum", "sum_only"):
        return GatedSumOnly(num_mod=num_mod, dims_mod=dims_mod, dim_out=dim_out, **kwargs)
    if variant in ("pairwise_only", "pairwise", "pair_only"):
        return PairwiseOnly(num_mod=num_mod, dims_mod=dims_mod, dim_out=dim_out, **kwargs)
    if variant in ("concat_mlp", "concat", "mlp"):
        return ConcatMLP(num_mod=num_mod, dims_mod=dims_mod, dim_out=dim_out, **kwargs)

    raise ValueError(
        f"Unknown multimodal fusion variant '{variant}'. "
        f"Valid options: 'glrx', 'uniform_avg', 'gated_sum', 'pairwise', 'concat_mlp'"
    )

