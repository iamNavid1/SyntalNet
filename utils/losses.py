import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")

class ClassBalancedFocalLoss(BaseLoss):
    """
    Class-balanced focal loss with optional label smoothing.

    :param counts: Number of samples for each class.
    :param beta: Hyperparameter controlling effective number of samples (see Cui et al.).
    :param gamma: Focal loss focusing parameter.
    :param smoothing: Amount of label smoothing to apply.
    :param mode: {'focal', 'sigmoid', 'softmax'}
    """

    def __init__(
        self,
        counts: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        smoothing: float = 0.0,
        mode: Literal["focal", "sigmoid", "softmax"] = "focal",
    ):
        super().__init__()
        counts = counts.float().clamp_min(1)
        effective_num = 1.0 - beta ** counts
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * counts.numel()
        self.register_buffer("alpha", weights)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.mode = mode
        self.num_classes = int(counts.numel())

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        # [N] -> [N, C]
        return F.one_hot(target, num_classes=self.num_classes).to(dtype=torch.float32, device=target.device)

    def _per_example_alpha(self, target: torch.Tensor) -> torch.Tensor:
        # Gather per-example class weight -> [N]
        return self.alpha.gather(0, target)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param logits: [N, C] float tensor
        :param target: [N] tensor with class indices
        :returns: Scalar loss (torch.Tensor)
        """
        assert logits.dim() == 2 and logits.size(1) == self.num_classes, "logits must be [N, C]"
        assert target.dim() == 1 and target.size(0) == logits.size(0), "target must be [N] matching batch size"

        N, C = logits.shape
        labels_one_hot = self._one_hot(target)  # [N, C]

        if self.smoothing > 0.0 and C > 1:
            eps = self.smoothing
            labels_one_hot = labels_one_hot * (1 - eps) + eps / (C - 1) * (1 - labels_one_hot)

        # Broadcast per-sample scalar weight of true class
        alpha_per_sample = self._per_example_alpha(target)         # [N]
        weights_matrix = alpha_per_sample.view(N, 1).expand(N, C)  # [N, C]

        if self.mode == "focal":
            # === Focal BCE with logits ===
            bce = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, reduction="none"
            )  # [N, C]

            if self.gamma == 0.0:
                modulator = torch.ones_like(logits)
            else:
                modulator = torch.exp(
                    -self.gamma * labels_one_hot * logits
                    - self.gamma * torch.log(1 + torch.exp(-logits))
                )

            loss = modulator * bce            # [N, C]
            weighted_loss = weights_matrix * loss  # [N, C]

            focal_loss = weighted_loss.sum()
            normalizer = labels_one_hot.sum().clamp_min(1.0)
            return focal_loss / normalizer

        elif self.mode == "sigmoid":
            # === Plain BCE-with-logits, one-vs-all ===
            return F.binary_cross_entropy_with_logits(
                input=logits,
                target=labels_one_hot,
                weight=weights_matrix,
                reduction="mean",
            )

        elif self.mode == "softmax":
            probs = logits.softmax(dim=1)
            return F.binary_cross_entropy(
                input=probs,
                target=labels_one_hot,
                weight=weights_matrix,
                reduction="mean",
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")
