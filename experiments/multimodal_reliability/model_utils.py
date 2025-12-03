from __future__ import annotations

import types
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

import models.builders as build
from experiments.multimodal_reliability.augmentations import ReliabilitySwitchAugmentor
from experiments.multimodal_reliability.config import (
    ExperimentConfig,
    FusionVariantConfig,
    ReliabilityAugmentationConfig,
)


class ReliabilityFusionWrapper(nn.Module):
    """
    Thin wrapper that injects the reliability-switch augmentation before calling the
    underlying multimodal fusion module.
    """

    def __init__(
        self,
        fusion_module: nn.Module,
        augmentor: Optional[ReliabilitySwitchAugmentor] = None,
    ):
        super().__init__()
        self.fusion = fusion_module
        self.augmentor = augmentor
        self._aug_enabled = augmentor is not None

    def enable_corruption(self, flag: bool) -> None:
        self._aug_enabled = flag and (self.augmentor is not None)

    def forward(self, z_list, *args, **kwargs):
        inputs = z_list
        if (
            self.training
            and self._aug_enabled
            and self.augmentor is not None
            and z_list
        ):
            inputs = self.augmentor(z_list)
        return self.fusion(inputs, *args, **kwargs)

    def get_aux(self):
        if hasattr(self.fusion, "get_aux"):
            return self.fusion.get_aux()
        return None


def load_base_model(base_cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model = build.model(base_cfg).to(device)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    return state


def rebuild_fusion(
    model: nn.Module,
    variant: FusionVariantConfig,
    shared_dim: int,
) -> nn.Module:
    if not hasattr(model, "branches") or len(model.branches) < 1:
        raise ValueError("Model has no branches to fuse.")
    fusion_module = build.multimodal_fusion(
        variant=variant.name,
        num_mod=len(model.branches),
        dims_mod=shared_dim,
        **(variant.kwargs or {}),
    )
    return fusion_module


def wrap_model_with_reliability(
    model: nn.Module,
    fusion_module: nn.Module,
    aug_cfg: ReliabilityAugmentationConfig,
) -> ReliabilityFusionWrapper:
    augmentor = ReliabilitySwitchAugmentor(aug_cfg)
    wrapper = ReliabilityFusionWrapper(fusion_module, augmentor)
    model.mm_fusion = wrapper
    return wrapper


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "mm_fusion") and model.mm_fusion is not None:
        for param in model.mm_fusion.parameters():
            param.requires_grad = True

    if getattr(model, "individual_classifier", None) is not None:
        for param in model.individual_classifier.parameters():
            param.requires_grad = True

    if getattr(model, "group_classifier", None) is not None:
        for param in model.group_classifier.parameters():
            param.requires_grad = True


def ensure_branches_stay_eval(model: nn.Module) -> None:
    """
    Override nn.Module.train so that frozen branches stay in eval mode, while
    the train/eval flag still propagates to fusion + classifiers.
    """
    if getattr(model, "_reliability_branch_hooked", False):
        return

    original_train = model.train

    def patched_train(self, mode: bool = True):
        result = original_train(mode)
        if mode and hasattr(self, "branches"):
            self.branches.eval()
        return result

    model.train = types.MethodType(patched_train, model)
    model._reliability_branch_hooked = True


def prepare_model(
    exp_cfg: ExperimentConfig,
    base_cfg: Dict[str, Any],
    ckpt_path: str,
    device: torch.device,
) -> Tuple[nn.Module, ReliabilityFusionWrapper]:
    model = load_base_model(base_cfg, device)
    load_checkpoint(model, ckpt_path, device)

    shared_dim = 128
    fusion_module = rebuild_fusion(model, exp_cfg.fusion, shared_dim)
    wrapper = wrap_model_with_reliability(model, fusion_module, exp_cfg.reliability)
    freeze_backbone(model)
    ensure_branches_stay_eval(model)
    model.to(device)
    return model, wrapper

