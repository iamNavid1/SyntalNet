import torch
from typing import Dict, Any
from io import StringIO
from rich.console import Console
from rich.tree import Tree
from rich.text import Text


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

    def _fmt(label: str, n: int) -> Text:
        return Text.from_markup(f"[bold]{label}[/bold] — {n:,}")

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, color_system=None)

    root = Tree(Text.from_markup(f"[bold]Model[/bold] [dim](total {total:,})"))

    # Branches and their intra-branch fusion modules
    if hasattr(model, "branches") and isinstance(model.branches, dict):
        for name, branch in model.branches.items():
            b_params = count_parameters(branch)
            b_node = root.add(_fmt(f"Branch '{name}'", b_params))
            if getattr(branch, "mc_fusion", None) is not None:
                mc_params = count_parameters(branch.mc_fusion)
                b_node.add(_fmt("mc_fusion", mc_params))
    else:
        if hasattr(model, "branch"):
            b_params = count_parameters(model.branch)
            root.add(_fmt(f"Branch '{getattr(model, 'branch_key', 'main')}'", b_params))

    # Multimodal fusion
    if getattr(model, "mm_fusion", None) is not None:
        mm_params = count_parameters(model.mm_fusion)
        root.add(_fmt("mm_fusion", mm_params))

    # Individual classifier and its sub-modules
    ind_cls = getattr(model, "individual_classifier", None)
    if ind_cls is not None:
        ind_node = root.add(_fmt("individual_classifier", count_parameters(ind_cls)))
        if hasattr(ind_cls, "trunk"):
            ind_node.add(_fmt("trunk", count_parameters(ind_cls.trunk)))
        adapters = getattr(ind_cls, "adapters", {}) or {}
        for h, adapter in adapters.items():
            ind_node.add(_fmt(f"adapter[{h}]", count_parameters(adapter)))
        classifiers = getattr(ind_cls, "classifiers", {}) or {}
        for h, clf in classifiers.items():
            ind_node.add(_fmt(f"classifier[{h}]", count_parameters(clf)))

    # Group classifier and its sub-modules
    grp_cls = getattr(model, "group_classifier", None)
    if grp_cls is not None:
        grp_node = root.add(_fmt("group_classifier", count_parameters(grp_cls)))
        if hasattr(grp_cls, "trunk"):
            grp_node.add(_fmt("trunk", count_parameters(grp_cls.trunk)))
        adapters = getattr(grp_cls, "adapters", {}) or {}
        for h, adapter in adapters.items():
            grp_node.add(_fmt(f"adapter[{h}]", count_parameters(adapter)))
        classifiers = getattr(grp_cls, "classifiers", {}) or {}
        for h, clf in classifiers.items():
            grp_node.add(_fmt(f"classifier[{h}]", count_parameters(clf)))

    console.print(root)
    logger.info("\n%s", buf.getvalue())


