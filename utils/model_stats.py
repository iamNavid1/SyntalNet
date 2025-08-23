import torch
from typing import Dict, Any


def count_parameters(module: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_training_hyperparams(cfg: Dict[str, Any], logger) -> None:
    args = cfg.get('training', {})
    if not logger or not args:
        return
    logger.info('Training hyperparameters:')
    for key, value in args.items():
        logger.info(f"  {key}: {value}")


def log_model_hyperparams(cfg: Dict[str, Any], logger) -> None:
    args = cfg.get('model', {}).get('args', {})
    if not logger or not args:
        return
    logger.info('Model hyperparameters:')
    for key, value in args.items():
        logger.info(f"  {key}: {value}")


def log_param_counts(model: torch.nn.Module, logger) -> None:
    if not logger:
        return
    total = count_parameters(model)
    logger.info(f"Total parameters: {total:,}")

    # Branches and their intra-branch fusion modules
    if hasattr(model, "branches"):
        for name, branch in model.branches.items():
            b_params = count_parameters(branch)
            logger.info(f"Branch '{name}': {b_params:,}")
            if getattr(branch, "mc_fusion", None) is not None:
                mc_params = count_parameters(branch.mc_fusion)
                logger.info(f"  mc_fusion: {mc_params:,}")

    # Multimodal fusion
    if getattr(model, "mm_fusion", None) is not None:
        logger.info(
            "mm_fusion: %s", f"{count_parameters(model.mm_fusion):,}"
        )

    # Individual classifier and its sub-modules
    ind_cls = getattr(model, "individual_classifier", None)
    if ind_cls is not None:
        logger.info(
            "individual_classifier: %s", f"{count_parameters(ind_cls):,}"
        )
        logger.info(
            "  trunk: %s", f"{count_parameters(ind_cls.trunk):,}"
        )
        for h, adapter in ind_cls.adapters.items():
            logger.info(
                "  adapter[%s]: %s", h, f"{count_parameters(adapter):,}"
            )
        for h, clf in ind_cls.classifiers.items():
            logger.info(
                "  classifier[%s]: %s", h, f"{count_parameters(clf):,}"
            )

    # Group classifier and its sub-modules
    grp_cls = getattr(model, "group_classifier", None)
    if grp_cls is not None:
        logger.info(
            "group_classifier: %s", f"{count_parameters(grp_cls):,}"
        )
        logger.info(
            "  trunk: %s", f"{count_parameters(grp_cls.trunk):,}"
        )
        for h, adapter in grp_cls.adapters.items():
            logger.info(
                "  adapter[%s]: %s", h, f"{count_parameters(adapter):,}"
            )
        for h, clf in grp_cls.classifiers.items():
            logger.info(
                "  classifier[%s]: %s", h, f"{count_parameters(clf):,}"
            )

