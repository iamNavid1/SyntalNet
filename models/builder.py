from __future__ import annotations

import yaml
import importlib
from typing import Any, Dict


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
    
    