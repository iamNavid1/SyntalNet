from __future__ import annotations

import yaml
import importlib
from typing import Any, Dict, List

from models.multichannel_fusion import BSC_X, BSXProjOnly, ConcatProjFusion, UniformAvgFusion

def build_model(config: Dict[str, Any]):
    cfg = config["model"]
    name = cfg["name"]
    args = cfg.get("args", {})

    module = importlib.import_module(f"models.{name}")
    model_cls = getattr(module, name)
    return model_cls(**args)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def build_multichannel_fusion(
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
    raise ValueError(f"Unknown BSCX variant '{variant}'")

