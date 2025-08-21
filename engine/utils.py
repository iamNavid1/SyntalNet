from typing import Dict
import torch


def BuildAutocastKWargs(cfg: Dict, device) -> Dict:
    cfg = cfg.get("training", {})
    use_amp = bool(cfg.get("amp", False))
    amp_dtype_str = str(cfg.get("amp_dtype", "")).lower()

    default_dtype = "bf16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "fp16"
    if amp_dtype_str not in {"bf16", "fp16"}:
        amp_dtype_str = default_dtype

    if device.type == "cpu" and amp_dtype_str == "fp16":
        amp_dtype_str = "bf16"

    amp_enabled = bool(use_amp and (device.type in {"cuda", "cpu"}))
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16

    if device.type == "cpu" and amp_dtype is torch.float16:
        amp_enabled = False

    return dict(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)


def _detach_tree(x):
    if torch.is_tensor(x): return x.detach()
    if isinstance(x, dict): return {k: _detach_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): 
        t = type(x)
        return t(_detach_tree(v) for v in x)
    return x
