import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


def compute_logit_adjustment(counts: torch.Tensor, tau: float, mode: str) -> torch.Tensor:
    """Compute logit adjustment vector from class counts.

    :param counts: per-class sample counts.
    :param tau: scaling factor for the adjustment.
    :param mode: {'focal', 'sigmoid', 'softmax'}
    :returns Logit adjustment tensor of shape [C].
    """
    priors = counts.float() / counts.sum()
    if tau == 0.0:
        return torch.zeros_like(priors)
    if mode == "multiclass":
        adj = torch.log(priors)
    else:
        adj = torch.log(priors / (1.0 - priors))
    return tau * adj


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-balanced focal loss with optional label smoothing and logit adjustment.

    :param counts: Number of samples for each class.
    :param beta: Hyperparameter controlling effective number of samples (see Cui et al.).
    :param gamma: Focal loss focusing parameter.
    :param smoothing: Amount of label smoothing to apply.
    :param mode: {'multiclass', 'multilabel'}
    """
    def __init__(
        self,
        counts: torch.Tensor,
        beta: float = 0.9995,
        gamma: float = 2.0,
        tau: float = 0.0,
        smoothing: float = 0.0,
        mode: Literal["multiclass", "multilabel"] = "multiclass",
    ):
        super().__init__()
        counts = counts.float().clamp_min(1)
        effective_num = 1.0 - beta**counts
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * counts.numel()
        self.register_buffer("alpha", weights)
        self.register_buffer("logit_adj", compute_logit_adjustment(counts, tau, mode))
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.mode = mode
        self.num_classes = int(counts.numel())

    def _per_example_alpha(self, target: torch.Tensor) -> torch.Tensor:
        return self.alpha.gather(0, target)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "multiclass":
            assert logits.dim() == 2 and logits.size(1) == self.num_classes, "logits must be [N, C]"
            assert target.dim() == 1 and target.size(0) == logits.size(0), "target must be [N] matching batch size"

            logits = logits + self.logit_adj.unsqueeze(0)

            ce = F.cross_entropy(
                logits,
                target,
                reduction="none",
                label_smoothing=self.smoothing,
            )

            pt = logits.softmax(dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
            pt = pt.clamp(1e-8, 1 - 1e-8)
            modulator = (1 - pt).pow(self.gamma) if self.gamma > 0.0 else torch.ones_like(pt)

            alpha = self._per_example_alpha(target)

            loss = alpha * modulator * ce

            return loss.sum() / alpha.sum()

        elif self.mode == "multilabel":
            assert logits.shape == target.shape, "target must be [N, C]"

            logits = logits + self.logit_adj.unsqueeze(0)
            labels = target.to(dtype=torch.float32)

            bce = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.alpha, reduction="none"
            )

            p = torch.sigmoid(logits)
            pt = labels * p + (1 - labels) * (1 - p)
            modulator = (1 - pt).pow(self.gamma) if self.gamma > 0.0 else torch.ones_like(pt)

            loss = modulator * bce

            return loss.mean()

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")



# class ClassBalancedFocalLoss(nn.Module):
#     """
#     Class-balanced focal loss with optional label smoothing.

#     :param counts: Number of samples for each class.
#     :param beta: Hyperparameter controlling effective number of samples (see Cui et al.).
#     :param gamma: Focal loss focusing parameter.
#     :param smoothing: Amount of label smoothing to apply.
#     :param mode: {'focal', 'sigmoid', 'softmax'}
#     """

#     def __init__(
#         self,
#         counts: torch.Tensor,
#         beta: float = 0.9999,
#         gamma: float = 2.0,
#         smoothing: float = 0.0,
#         mode: Literal["focal", "sigmoid", "softmax"] = "focal",
#         tau: float = 0.0,
#     ):
#         super().__init__()
#         counts = counts.float().clamp_min(1)
#         effective_num = 1.0 - beta ** counts
#         weights = (1.0 - beta) / effective_num
#         weights = weights / weights.sum() * counts.numel()
#         self.register_buffer("alpha", weights)
#         self.register_buffer("logit_adj", compute_logit_adjustment(counts, tau, mode))
#         self.beta = float(beta)
#         self.gamma = float(gamma)
#         self.smoothing = float(smoothing)
#         self.mode = mode
#         self.num_classes = int(counts.numel())

#     def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
#         # [N] -> [N, C]
#         return F.one_hot(target, num_classes=self.num_classes).to(dtype=torch.float32, device=target.device)

#     def _per_example_alpha(self, target: torch.Tensor) -> torch.Tensor:
#         # Gather per-example class weight -> [N]
#         return self.alpha.gather(0, target)

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         :param logits: [N, C] float tensor
#         :param target: [N] tensor with class indices
#         :returns: Scalar loss (torch.Tensor)
#         """
#         assert logits.dim() == 2 and logits.size(1) == self.num_classes, "logits must be [N, C]"
#         assert target.dim() == 1 and target.size(0) == logits.size(0), "target must be [N] matching batch size"

#         logits = logits + self.logit_adj.unsqueeze(0)

#         N, C = logits.shape
#         labels_one_hot = self._one_hot(target)  # [N, C]

#         if self.smoothing > 0.0 and C > 1:
#             eps = self.smoothing
#             labels_one_hot = labels_one_hot * (1 - eps) + eps / (C - 1) * (1 - labels_one_hot)

#         # Broadcast per-sample scalar weight of true class
#         alpha_per_sample = self._per_example_alpha(target)         # [N]
#         weights_matrix = alpha_per_sample.view(N, 1).expand(N, C)  # [N, C]

#         if self.mode == "focal":
#             # === Focal BCE with logits ===
#             bce = F.binary_cross_entropy_with_logits(
#                 input=logits, target=labels_one_hot, reduction="none"
#             )  # [N, C]

#             if self.gamma == 0.0:
#                 modulator = torch.ones_like(logits)
#             else:
#                 modulator = torch.exp(
#                     -self.gamma * labels_one_hot * logits
#                     - self.gamma * F.softplus(-logits)
#                 )

#             loss = modulator * bce            # [N, C]
#             weighted_loss = weights_matrix * loss  # [N, C]

#             focal_loss = weighted_loss.sum()
#             normalizer = labels_one_hot.sum().clamp_min(1.0)
#             return focal_loss / normalizer

#         elif self.mode == "sigmoid":
#             # === Plain BCE-with-logits, one-vs-all ===
#             return F.binary_cross_entropy_with_logits(
#                 input=logits,
#                 target=labels_one_hot,
#                 weight=weights_matrix,
#                 reduction="mean",
#             )

#         elif self.mode == "softmax":
#             probs = logits.softmax(dim=1)
#             return F.binary_cross_entropy(
#                 input=probs,
#                 target=labels_one_hot,
#                 weight=weights_matrix,
#                 reduction="mean",
#             )

#         else:
#             raise ValueError(f"Unknown mode: {self.mode!r}")





class ClassBalancedCELoss(nn.Module):
    """Class-balanced cross entropy with optional label smoothing and logit adjustment."""
    def __init__(
        self, 
        counts: torch.Tensor, 
        beta: float = 0.9995, 
        tau: float = 1.0, 
        smoothing: float = 0.0
    ):
        super().__init__()
        counts = counts.float().clamp_min(1)
        effective_num = 1.0 - beta**counts
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * counts.numel()
        self.register_buffer("alpha", weights)
        self.register_buffer("logit_adj", compute_logit_adjustment(counts, tau, mode="softmax"))
        self.smoothing = float(smoothing)
        self.num_classes = int(counts.numel())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 2 and logits.size(1) == self.num_classes
        assert target.dim() == 1 and target.size(0) == logits.size(0)

        logits = logits + self.logit_adj.unsqueeze(0)

        if self.smoothing > 0.0 and self.num_classes > 1:
            eps = self.smoothing
            labels = F.one_hot(target, num_classes=self.num_classes).to(dtype=torch.float32, device=target.device)
            labels = labels * (1 - eps) + eps / (self.num_classes - 1) * (1 - labels)
            logp = F.log_softmax(logits, dim=1)
            per_example = -(labels * logp).sum(dim=1)
            w = self.alpha.gather(0, target)
            loss = per_example * w
            return loss.sum() / w.sum()

        return F.cross_entropy(logits, target, weight=self.alpha, reduction="mean")
