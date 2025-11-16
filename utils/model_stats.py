from __future__ import annotations
import re
from io import StringIO
from typing import Any, Dict, Iterable, Optional, Tuple, List

import torch.nn as nn
from rich.console import Console
from rich.text import Text
from rich.tree import Tree


def count_parameters(module: Optional[nn.Module], trainable_only: bool = True) -> int:
    """Return number of params; 0 for None. Accepts Modules only."""
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


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


def _fmt_label(label: str, n: int) -> Text:
    return Text.from_markup(f"[bold]{label}[/bold] — {n:,}")


def _children(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    # named_children() never returns None children; safe.
    return list(module.named_children())

def _should_include(name: str,
                    include_re: Optional[re.Pattern],
                    exclude_re: Optional[re.Pattern]) -> bool:
    if include_re and not include_re.search(name):
        return False
    if exclude_re and exclude_re.search(name):
        return False
    return True

def log_param_counts_detailed(
    model: nn.Module,
    logger,
    *,
    trainable_only: bool = True,
    max_depth: Optional[int] = None,
    min_params: int = 0,
    topk_per_level: Optional[int] = None,
    include_name_regex: Optional[str] = None,
    exclude_name_regex: Optional[str] = None,
    show_leaf_shapes: bool = False,
    show_type: bool = False,
) -> None:
    if not logger:
        return

    root_mod = getattr(model, "module", model)
    total = count_parameters(root_mod, trainable_only=trainable_only)

    include_re = re.compile(include_name_regex) if include_name_regex else None
    exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None

    # cache param counts
    param_cache: dict[int, int] = {}

    def mod_params(mod: Optional[nn.Module]) -> int:
        if mod is None:
            return 0
        mid = id(mod)
        if mid not in param_cache:
            param_cache[mid] = count_parameters(mod, trainable_only=trainable_only)
        return param_cache[mid]

    def leaf_suffix(mod: nn.Module) -> str:
        if not show_leaf_shapes:
            return ""
        parts = []
        for n, p in mod.named_parameters(recurse=False):
            if trainable_only and not p.requires_grad:
                continue
            parts.append(f"{n}:{tuple(p.shape)}")
        return ("  [" + ", ".join(parts) + "]") if parts else ""

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, color_system=None)

    title = f"[bold]Model[/bold] [dim](total {total:,})"
    tree = Tree(Text.from_markup(title))

    def add_node(node: Tree, mod: nn.Module, path: str, depth: int) -> None:
        if not _should_include(path, include_re, exclude_re):
            return
        n_params = mod_params(mod)
        if n_params < min_params and depth > 0:
            return

        label = path.split(".")[-1] or path
        if show_type:
            label = f"{label} : {mod.__class__.__name__}"
        node_here = node.add(_fmt_label(label + leaf_suffix(mod), n_params))

        if max_depth is not None and depth >= max_depth:
            return

        kids = _children(mod)
        if not kids:
            return

        kids_sorted = sorted(kids, key=lambda kv: mod_params(kv[1]), reverse=True)
        if topk_per_level is not None:
            kids_sorted = kids_sorted[:topk_per_level]
        for child_name, child_mod in kids_sorted:
            add_node(node_here, child_mod, f"{path}.{child_name}" if path else child_name, depth + 1)

    # Prefer grouping by major blocks, skip ones that are None.
    major_added = False
    for major_name in ("branches", "mm_fusion", "individual_classifier", "group_classifier"):
        major = getattr(root_mod, major_name, None)
        if major is None:
            continue
        if isinstance(major, nn.Module):
            n = mod_params(major)
            if n >= min_params and _should_include(major_name, include_re, exclude_re):
                add_node(tree, major, major_name, depth=1)
                major_added = True
        elif isinstance(major, (nn.ModuleDict, nn.Sequential, nn.ModuleList)):
            pass

    if not major_added:
        add_node(tree, root_mod, path="", depth=0)

    console.print(tree)
    logger.info("\n%s", buf.getvalue())


def log_param_counts(model: nn.Module, logger) -> None:
    log_param_counts_detailed(
        model, logger,
        trainable_only=True,
        max_depth=5,
        min_params=500,
        topk_per_level=10,
        include_name_regex=None,
        exclude_name_regex=None,
        show_leaf_shapes=True,
        show_type=True,
    )
