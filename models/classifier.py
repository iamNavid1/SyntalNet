import math
from turtle import forward

from sympy import group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Dict

from models.utils import MLP, GatedMLP, LoRA


class CosineProtoClassifier(nn.Module):
    """
    Per-label K-class cosine classifier with optional EMA prototypes and per-class fusion.
    Safe features:
      - temperature scaling
      - warm-up for prototype fusion
      - mask unseen prototypes early
    """
    def __init__(
        self,
        dim: int, 
        num_classes: int = 3,
        init_scale_param: float = 16.0,
        use_prototypes: bool = True,
        init_scale_proto: float = 8.0,
        proto_momentum: float = 0.96,
        fuse_init: float | List[float] = [-1.734, -1.386, -1.734],
        warmup_epochs: int = 5,
        learn_temperature: bool = True,
        init_temperature: float = 1.2,
    ):
        super().__init__()
        self.D = dim
        self.K = num_classes

        # parametric cosine classifier
        self.W = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.normal_(self.W, std=0.02)
        self.log_scale_param = nn.Parameter(torch.log(torch.tensor(init_scale_param)))

        # prototype cosine classifier
        self.use_prototypes = use_prototypes
        self.m = proto_momentum
        self.warmup_epochs = warmup_epochs
        if use_prototypes:
            self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, dim), dim=1))
            self.register_buffer("proto_counts", torch.zeros(num_classes))
            self.log_scale_proto = nn.Parameter(torch.log(torch.tensor(init_scale_proto)))
            if isinstance(fuse_init, (float, int)):
                fuse_init = torch.full((num_classes,), fuse_init)  # per-class λ in (0,1)
            else:
                fuse_init = torch.as_tensor(fuse_init, dtype=torch.float32)
                if fuse_init.numel() != num_classes:
                    raise ValueError(f"fuse_init must be scalar or length {num_classes}")
            self.fuse_logit = nn.Parameter(fuse_init)  # per-class λ logits

        # temperature
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.log_T = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer("T_buffer", torch.tensor(init_temperature))
            self.log_T = None

    @staticmethod
    def _cos_logits(z: torch.Tensor, W: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        W = F.normalize(W, dim=-1)
        return scale * (z @ W.t())  # (B,K)

    @torch.no_grad()
    def update_prototypes(self, z: torch.Tensor, y: torch.Tensor):
        if not self.use_prototypes:
            return
        
        z = F.normalize(z.float(), dim=-1)
        y = y.long()
        K, D = self.K, self.D

        # local class sums & counts
        mask = F.one_hot(y, num_classes=K).to(z.dtype)
        sums = mask.t().matmul(z)
        cnts = mask.sum(dim=0)

        # all-reduce to GLOBAL sums/cnts
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnts, op=dist.ReduceOp.SUM)

        # EMA update
        nz = (cnts > 0)
        if nz.any():
            means = sums[nz] / cnts[nz].unsqueeze(1).clamp_min(1e-6)
            proto = self.prototypes[nz]
            updated = self.m * proto + (1.0 - self.m) * means
            self.prototypes[nz] = F.normalize(updated, dim=1).to(self.prototypes.dtype)
        self.proto_counts += cnts.to(self.proto_counts.dtype)

    @staticmethod
    def _warmup_weight(progress: float) -> float:
        p = max(0.0, min(1.0, progress))
        w = 0.5 * (1.0 - math.cos(math.pi * p))
        return w

    def forward(self, z: torch.Tensor, epoch: Optional[int] = None, progress: Optional[float] = None) -> torch.Tensor:
        s_param = torch.exp(self.log_scale_param).clamp(5., 40.)
        logits_param = self._cos_logits(z, self.W, s_param)

        if self.use_prototypes:
            if progress is not None and self.warmup_epochs > 0:
                p = progress
            elif epoch is not None and self.warmup_epochs > 0:
                p = (epoch + 1.0) / float(self.warmup_epochs)
            else:
                p = 1.0
            w = self._warmup_weight(p)

            # compute prototype scale and logits
            s_proto = torch.exp(self.log_scale_proto).clamp(5., 50.)
            s_proto = s_proto * max(1e-6, w)  # ramp prototype scale
            logits_proto = self._cos_logits(z, self.prototypes, s_proto)

            # mask unseen classes for prototypes
            seen = (self.proto_counts > 0).float()
            if (seen == 0).any():
                logits_proto = logits_proto * seen.unsqueeze(0) + logits_param * (1 - seen).unsqueeze(0)

            # learned per-class fuse weight with warmup gate
            lam = torch.sigmoid(self.fuse_logit)  # (K,)
            lam = lam * torch.as_tensor(w, dtype=lam.dtype, device=lam.device)

            logits = (1 - lam.unsqueeze(0)) * logits_param + lam.unsqueeze(0) * logits_proto
        else:
            logits = logits_param

        # add temperature
        T = (torch.exp(self.log_T) if self.log_T is not None else self.T_buffer).clamp(1.0, 2.0)
        
        # store auxiliary data for tracking
        if self.use_prototypes:
            self._last_aux = {
                "fuse_logit_sigmoid": torch.sigmoid(self.fuse_logit).detach(),  # (K,)
                "warmup_weight": torch.tensor(w, device=self.fuse_logit.device).detach(),
            }
        else:
            self._last_aux = None
            
        return logits / T

    def get_aux(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return auxiliary data for tracking."""
        return self._last_aux


class DeepSetsTrunk(nn.Module):
    def __init__(
        self,
        dim: int,
        d_phi_h: int,
        d_phi: int,
        trunk_hidden: int,
        trunk_gate: str = "swiglu",
        p_drop_trunk: float = 0.15,
    ):
        super().__init__()
        self.phi = MLP(dim, d_phi_h, d_phi_h, p_drop=0.1)
        self.trunk = GatedMLP(
            in_features=d_phi,
            hidden_features=trunk_hidden,
            out_features=dim,
            p_drop=p_drop_trunk,
            gate_type=trunk_gate,
        )

    def forward(self, z_person):
        """
        z_person: (B, P, D)
        """
        B, P, D = z_person.shape
        z_person = z_person.view(B*P, D)
        h = self.phi(z_person).view(B, P, -1)
        s = h.sum(dim=1) / P
        z_grp = self.trunk(s)
        return z_grp


class ClassificationHead(nn.Module):
    """
    Shared trunk + per-label adapters + classifiers (cosine-prototype or MLP).
    """
    def __init__(
        self,
        dim: int,
        head_names: List[str],
        num_classes: int = 3,
        trunk_phi_h: Optional[int] = None, 
        trunk_phi: Optional[int] = None, 
        trunk_hidden: Optional[int] = None, 
        p_drop_trunk: float = 0.15,
        lora_adapter: bool = False,
        adapter_hidden: Optional[int] = None,
        p_drop_adapter: float = 0.15,
        residual: bool = True,
        use_prototypes: bool = True, 
        proto_momentum: float = 0.99, 
        warmup_epochs: int = 0,
        group_wise: bool = False,
        person_per_grp: int = 3,
        classifier_type: str = "cosine",
    ):
        super().__init__()
        self.heads = head_names
        self.P = person_per_grp
        
        self.group_wise = group_wise
        if group_wise:
            dim = dim * person_per_grp
        
        self.residual = residual
        if residual and not (group_wise and lora_adapter):
            self.alpha = nn.Parameter(torch.full((dim,), 2.5e-1))
        
        if group_wise:
            self.trunk = DeepSetsTrunk(dim, trunk_phi_h, trunk_phi, trunk_hidden, p_drop_trunk=p_drop_trunk)
        else:
            self.trunk = MLP(dim, trunk_hidden, dim, p_drop=p_drop_trunk)

        self.lora_adapter = lora_adapter
        if lora_adapter:
            self.adapters = nn.ModuleDict({name: LoRA(dim, adapter_hidden, use_film=True)
                                        for name in self.heads})
        else:
            self.adapters = nn.ModuleDict({name: MLP(dim, adapter_hidden, p_drop=p_drop_adapter)
                                        for name in self.heads})

        self.classifier_type = classifier_type
        if classifier_type == "cosine":
            self.classifiers = nn.ModuleDict({
                name: CosineProtoClassifier(
                    dim,
                    num_classes=num_classes,
                    use_prototypes=use_prototypes,
                    proto_momentum=proto_momentum,
                    warmup_epochs=warmup_epochs
                )
                for name in self.heads
            })
        elif classifier_type == "mlp":
            self.classifiers = nn.ModuleDict({
                name: nn.Linear(dim, num_classes)
                for name in self.heads
            })
        else:
            raise ValueError("classifier_type must be 'cosine' or 'mlp'")

    @torch.no_grad()
    def update_prototypes(self, z_dict: Dict[str, torch.Tensor], y_dict: Dict[str, torch.Tensor]):
        if self.classifier_type != "cosine":
            return
        for name in self.heads:
            if self.group_wise:
                B, D = z_dict[name].shape
                B = B // self.P
                z_dict[name] = z_dict[name].view(B, self.P, D).reshape(B, self.P * D)
            self.classifiers[name].update_prototypes(z_dict[name], y_dict[name])

    def forward(self, z: torch.Tensor, epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self.group_wise:
            B, D = z.shape
            B = B // self.P
            z = z.view(B, self.P, D)
        
        z_shared = self.trunk(z)
        
        if self.residual and not self.group_wise:
            z_shared = z + self.alpha * z_shared 

        z_head: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        for name in self.heads:
            z_out = self.adapters[name](z_shared)
            if self.residual and not self.lora_adapter:
                z_out = z_shared + self.alpha * z_out
            z_head[name] = z_out
            if self.classifier_type == "cosine":
                logits[name] = self.classifiers[name](z_out, epoch=epoch)
            else:
                logits[name] = self.classifiers[name](z_out)

        features: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            'shared': z_shared,
            'per_head': z_head,
        }
        return features, logits

