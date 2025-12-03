from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import yaml


@dataclass
class FusionVariantConfig:
    """Descriptor for the multimodal fusion variant to evaluate."""

    name: str = "glrx"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityAugmentationConfig:
    """Configuration for the reliability-switch corruption applied during training."""

    p_corrupt: float = 0.7
    dropout_range: Tuple[float, float] = (0.2, 0.5)
    noise_range: Tuple[float, float] = (0.2, 0.5)
    shuffle_range: Tuple[float, float] = (0.2, 0.5)
    mix: Dict[str, float] = field(
        default_factory=lambda: {"dropout": 0.4, "noise": 0.4, "shuffle": 0.2}
    )


@dataclass
class TrainingSettings:
    """Fine-tuning hyper-parameters for the reliability run."""

    num_epochs: int = 30
    val_interval: int = 0
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    grad_clip: Optional[float] = None
    accum_steps: Optional[int] = None
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_threshold: float = 0.01
    scheduler_min_lr: float = 1e-5


@dataclass
class StressTestSettings:
    """Post-hoc stress test sweeps."""

    corruption_grid: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "dropout": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
            "noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
            "shuffle": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
        }
    )
    per_modality: bool = True
    run_allocation: bool = True
    allocation_noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )


@dataclass
class ExperimentConfig:
    """Top-level configuration for the reliability experiment."""

    base_config_path: str
    base_checkpoint_dir: str
    base_checkpoint_name: str = "epoch_100.pth"
    cv_mode: str = "kfold"
    n_folds: int = 5
    output_root: str = "experiments/multimodal_reliability/resultsNEW"
    device: str = "cuda:0"
    fusion: FusionVariantConfig = field(default_factory=FusionVariantConfig)
    reliability: ReliabilityAugmentationConfig = field(
        default_factory=ReliabilityAugmentationConfig
    )
    training: TrainingSettings = field(default_factory=TrainingSettings)
    stress: StressTestSettings = field(default_factory=StressTestSettings)

    def checkpoint_for_fold(self, fold_idx: int) -> str:
        """Resolve the pretrained SyntalNet checkpoint for a fold."""
        fold_tag = f"fold_{fold_idx:02d}"
        return os.path.join(
            self.base_checkpoint_dir,
            self.cv_mode,
            fold_tag,
            self.base_checkpoint_name,
        )

    def fold_output_dir(self, variant_name: str, fold_idx: int) -> str:
        """Directory to store artifacts for a (variant, fold) pair."""
        fold_tag = f"fold_{fold_idx:02d}"
        return os.path.join(
            self.output_root,
            self.cv_mode,
            variant_name.lower(),
            fold_tag,
        )


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty config file: {path}")
    return data


def _build_dataclass(cls, payload: Optional[Dict[str, Any]]):
    payload = payload or {}
    return cls(**payload)


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load an experiment config YAML into structured dataclasses."""
    raw = _load_yaml(path)

    fusion = _build_dataclass(
        FusionVariantConfig,
        raw.get("fusion") or {"name": raw.get("fusion_variant", "glrx"),
                              "kwargs": raw.get("fusion_kwargs", {})},
    )

    cfg = ExperimentConfig(
        base_config_path=raw["base_config_path"],
        base_checkpoint_dir=raw["base_checkpoint_dir"],
        base_checkpoint_name=raw.get("base_checkpoint_name", "epoch_100.pth"),
        cv_mode=raw.get("cv_mode", "kfold"),
        n_folds=int(raw.get("n_folds", 5)),
        output_root=raw.get(
            "output_root", "experiments/multimodal_reliability/resultsNEW"
        ),
        device=raw.get("device", "cuda:0"),
        fusion=fusion,
        reliability=_build_dataclass(
            ReliabilityAugmentationConfig, raw.get("reliability")
        ),
        training=_build_dataclass(TrainingSettings, raw.get("training")),
        stress=_build_dataclass(StressTestSettings, raw.get("stress")),
    )
    return cfg

