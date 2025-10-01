from typing import Dict
import numpy as np
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


BRANCH_MODALITY_MAP = {
    "videokinetic": {
        "feat": ["face", "pose"],
        "emb": "video",
    },
    "dialogue": {
        "feat": ["turns"],
        "emb": "utterance",
    },
    "acoustic": {
        "feat": ["sentiment", "prosody"],
        "emb": "audio",
    },
}


def modalities_to_branches(batch_data: Dict[str, tuple]) -> Dict[str, Dict[str, tuple]]:
    """
    Group modality tensors into model branches.
    """
    branch_data: Dict[str, Dict[str, tuple]] = {}

    for branch, spec in BRANCH_MODALITY_MAP.items():
        mod_dict: Dict[str, tuple] = {}

        for modality in spec["feat"]:
            if modality in batch_data:
                mod_dict[modality] = batch_data[modality]

        emb_mod = spec["emb"]
        if emb_mod in batch_data:
            mod_dict[emb_mod] = batch_data[emb_mod]

        if mod_dict:  # skip branches with no available modalities in this batch
            branch_data[branch] = mod_dict

    return branch_data


def format_metrics(metrics: Dict[str, Dict]) -> str:
    def _format(d: Dict, indent: int) -> list[str]:
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(" " * indent + f"{key}:")
                lines.extend(_format(value, indent + 2))
            else:
                if isinstance(value, np.ndarray):
                    arr = np.array2string(value, separator=", ")
                    arr = arr.replace("\n", "\n" + " " * (indent + 2))
                    value_str = arr
                else:
                    value_str = str(value)
                lines.append(" " * indent + f"{key}: {value_str}")
        return lines

    if isinstance(metrics, str):
        return metrics
    return "\n".join(_format(metrics, 0))
