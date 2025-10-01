import re
from typing import Any, Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

def _bytes_of_tensor(t: torch.Tensor) -> int:
    return 0 if t is None else t.numel() * t.element_size()

def _bytes_of(obj: Any) -> int:
    # Recursively sum tensor storage in nested outputs (Tensor, tuple, list, dict, etc.)
    if torch.is_tensor(obj):
        return _bytes_of_tensor(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_bytes_of(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_bytes_of(v) for v in obj.values())
    return 0

def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes/1024**3:.2f} GB"
    if nbytes >= 1024**2:
        return f"{nbytes/1024**2:.2f} MB"
    if nbytes >= 1024:
        return f"{nbytes/1024:.2f} KB"
    return f"{nbytes} B"

class ActivationMemoryProfiler:
    """
    Attach to selected modules and record:
      - output activation bytes (forward returns)
      - saved-for-backward bytes (true autograd footprint)
      - grad-output bytes (backward)
    Works with modules that return tuples (e.g., (x, m)).
    """

    def __init__(
        self,
        model: nn.Module,
        include_classes: Optional[Sequence[Type[nn.Module]]] = None,
        include_name_regex: Optional[str] = None,
        exclude_name_regex: Optional[str] = None,
    ):
        self.model = model
        self.include_classes = tuple(include_classes) if include_classes else None
        self.include_name_re = re.compile(include_name_regex) if include_name_regex else None
        self.exclude_name_re = re.compile(exclude_name_regex) if exclude_name_regex else None

        # stats[name] = dict(calls, out_bytes, saved_bytes, grad_out_bytes)
        self.stats: Dict[str, Dict[str, int]] = {}
        self._handles: List[Any] = []
        self._wrapped_forwards: Dict[nn.Module, Any] = {}
        self._has_saved_hooks = hasattr(torch.autograd.graph, "saved_tensors_hooks")

        self._register()

    def _want(self, name: str, mod: nn.Module) -> bool:
        if self.include_classes and not isinstance(mod, self.include_classes):
            return False
        if self.include_name_re and not self.include_name_re.search(name):
            return False
        if self.exclude_name_re and self.exclude_name_re.search(name):
            return False
        return True

    def _ensure_entry(self, name: str):
        if name not in self.stats:
            self.stats[name] = dict(calls=0, out_bytes=0, saved_bytes=0, grad_out_bytes=0)

    def _register_module(self, name: str, mod: nn.Module):
        self._ensure_entry(name)

        # 1) Wrap forward to measure outputs + saved-for-backward
        orig_forward = mod.forward

        def wrapped_forward(*args, **kwargs):
            saved_local = 0

            if self._has_saved_hooks:
                def pack_hook(t):
                    nonlocal saved_local
                    saved_local += _bytes_of_tensor(t)
                    return t
                def unpack_hook(t):
                    return t
                ctx = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
            else:
                # Fallback: no saved-tensor visibility; we still log outputs
                class _DummyCtx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                ctx = _DummyCtx()

            with ctx:
                out = orig_forward(*args, **kwargs)

            out_bytes = _bytes_of(out)
            s = self.stats[name]
            s["calls"] += 1
            s["out_bytes"] += int(out_bytes)
            s["saved_bytes"] += int(saved_local)
            return out

        self._wrapped_forwards[mod] = mod.forward
        mod.forward = wrapped_forward  # monkey-patch

        # 2) Backward hook to measure grad-output bytes
        def bwd_hook(module, grad_input, grad_output):
            # grad_output is a tuple
            gbytes = _bytes_of(grad_output)
            self.stats[name]["grad_out_bytes"] += int(gbytes)
            return None

        self._handles.append(mod.register_full_backward_hook(bwd_hook))

    def _register(self):
        for name, mod in self.model.named_modules():
            if name == "" or mod is self.model:
                continue
            if self._want(name, mod):
                self._register_module(name, mod)

    def clear(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        # restore forwards
        for mod, f in self._wrapped_forwards.items():
            mod.forward = f
        self._wrapped_forwards.clear()

    def summary(self, sort_by: str = "saved_bytes", topk: Optional[int] = None) -> str:
        assert sort_by in {"saved_bytes", "out_bytes", "grad_out_bytes", "total"}, "invalid sort key"
        rows = []
        for name, s in self.stats.items():
            total = s["saved_bytes"] + s["out_bytes"]
            rows.append((
                name, s["calls"], s["out_bytes"], s["saved_bytes"], s["grad_out_bytes"], total
            ))
        key_idx = {"out_bytes":2, "saved_bytes":3, "grad_out_bytes":4, "total":5}[sort_by]
        rows.sort(key=lambda x: x[key_idx], reverse=True)
        if topk is not None:
            rows = rows[:topk]

        # build text table
        lines = []
        lines.append(f"{'module':60}  {'calls':>5}  {'out':>10}  {'saved':>10}  {'grad_out':>10}  {'total':>10}")
        for name, calls, out_b, saved_b, grad_b, total_b in rows:
            lines.append(
                f"{name[:60]:60}  {calls:5d}  {_fmt_bytes(out_b):>10}  {_fmt_bytes(saved_b):>10}  "
                f"{_fmt_bytes(grad_b):>10}  {_fmt_bytes(total_b):>10}"
            )
        return "\n".join(lines)

    def print_summary(self, **kwargs):
        print(self.summary(**kwargs))
