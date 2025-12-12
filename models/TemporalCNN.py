import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from data.transforms import EMBEDDING_DIMS
from engine.utils import BRANCH_MODALITY_MAP
from models.base_model import BaseModel


FEATURE_DIMS = {
    "face": 212,
    "pose": 182,
    "turns": 12,
    "sentiment": 5,
    "prosody": 7,
}


class _MaskedTemporalCNN(nn.Module):
    """Simple temporal CNN encoder for per-person sequences."""

    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build network
        layers: List[nn.Module] = []
        c_in = in_channels
        for _ in range(self.num_layers):
            conv = nn.Conv1d(
                in_channels=c_in,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )
            layers.extend([
                conv,
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.hidden_size),
            ])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            c_in = self.hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, P, T, F)
            mask: Optional tensor with same shape as x marking valid entries.
        Returns:
            Tensor of shape (B, P, hidden_size)
        """
        B, P, T, F = x.shape
        mask_1d: Optional[torch.Tensor] = None
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.dim() == x.dim() - 1:
                mask = mask.unsqueeze(-1)
            if mask.shape[-1] != 1:
                if mask.shape[-1] == F:
                    mask = mask[..., :1]
                else:
                    mask = mask[..., :1]
            if mask.shape[-2] != T:
                target_len = T
                if mask.shape[-2] > target_len:
                    mask = mask[..., :target_len, :]
                else:
                    pad_shape = mask.shape[:-2] + (target_len - mask.shape[-2], mask.shape[-1])
                    pad = torch.ones(pad_shape, device=x.device, dtype=x.dtype)
                    mask = torch.cat([mask, pad], dim=-2)
            mask = mask.clamp_(0.0, 1.0)
            x = x * mask
            mask_1d = mask.view(B * P, T)
        x = x.view(B * P, T, F)
        x = x.permute(0, 2, 1)  # (B*P, F, T)

        x = self.net(x)

        if mask_1d is not None:
            mask_1d = mask_1d.unsqueeze(1)  # (B*P, 1, T)
            denom = mask_1d.sum(dim=-1).clamp_min(1.0)
            x = (x * mask_1d).sum(dim=-1) / denom
        else:
            x = x.mean(dim=-1)

        x = x.view(B, P, -1)
        return x


class _SimpleClassifier(nn.Module):
    """Lightweight MLP classifier used by the baselines."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_names: Tuple[str, ...],
        dropout: float = 0.1,
        group_wise: bool = False,
    ) -> None:
        super().__init__()
        self.heads = list(head_names)
        self.group_wise = group_wise
        self.hidden_dim = hidden_dim
        self.classifier_type = "mlp"
        self.K = 3

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.classifiers = nn.ModuleDict()
        for name in self.heads:
            head = nn.Linear(hidden_dim, self.K)
            head.K = self.K  # for validator compatibility
            self.classifiers[name] = head

    def forward(
        self, z: torch.Tensor, epoch: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor | Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self.group_wise:
            if z.dim() == 3:
                z = z.mean(dim=1)
            shared = self.shared(z)
            per_head = {name: shared for name in self.heads}
            logits = {name: classifier(shared) for name, classifier in self.classifiers.items()}
            return {"shared": shared, "per_head": per_head}, logits

        B, P, D = z.shape
        flat = z.view(B * P, D)
        shared_flat = self.shared(flat)
        shared = shared_flat.view(B, P, -1)
        per_head = {name: shared for name in self.heads}
        logits = {name: classifier(shared_flat) for name, classifier in self.classifiers.items()}
        return {"shared": shared, "per_head": per_head}, logits

    @torch.no_grad()
    def update_prototypes(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op for the CNN baseline
        return None


class TemporalCNN(BaseModel):
    """Baseline multimodal model with temporal CNN encoders."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_cnn_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        ind_cls_heads: Tuple[str, ...] = ("Engagement", "Lead"),
        grp_cls_heads: Tuple[str, ...] = ("Synchrony", "Confidence", "Transition"),
        classifier_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.branch_names = tuple(BRANCH_MODALITY_MAP.keys())
        self.branches = nn.ModuleDict()
        self.feature_dim_to_mod = {
            FEATURE_DIMS[name]: name for name in FEATURE_DIMS
        }

        modality_names = sorted(
            {
                mod
                for spec in BRANCH_MODALITY_MAP.values()
                for mod in [*spec["feat"], spec["emb"]]
                if mod is not None
            }
        )
        
        # Build encoders
        self.sequence_encoders = nn.ModuleDict()
        for mod in modality_names:
            # Determine input dimension for this modality
            if mod in FEATURE_DIMS:
                in_dim = FEATURE_DIMS[mod]
            elif mod in EMBEDDING_DIMS:
                in_dim = EMBEDDING_DIMS[mod]
            else:
                raise ValueError(f"Unknown modality '{mod}' - not in FEATURE_DIMS or EMBEDDING_DIMS")
            
            self.sequence_encoders[mod] = _MaskedTemporalCNN(
                hidden_size=hidden_size,
                in_channels=in_dim,
                num_layers=num_cnn_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        self.branch_projectors = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for branch in self.branch_names
            }
        )

        self.backbone_norm = nn.LayerNorm(hidden_size * len(self.branch_names))
        self.backbone_dropout = nn.Dropout(dropout)

        classifier_hidden = classifier_hidden or hidden_size
        self.individual_classifier: Optional[_SimpleClassifier] = None
        self.group_classifier: Optional[_SimpleClassifier] = None

        feature_dim = hidden_size * len(self.branch_names)
        if ind_cls_heads:
            self.individual_classifier = _SimpleClassifier(
                input_dim=feature_dim,
                hidden_dim=classifier_hidden,
                head_names=ind_cls_heads,
                dropout=dropout,
                group_wise=False,
            )
        if grp_cls_heads:
            self.group_classifier = _SimpleClassifier(
                input_dim=feature_dim,
                hidden_dim=classifier_hidden,
                head_names=grp_cls_heads,
                dropout=dropout,
                group_wise=True,
            )

    @torch.no_grad()
    def update_prototypes(
        self,
        z_dict: Dict[torch.Tensor | str, Dict[str, torch.Tensor]],
        y_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        if self.individual_classifier is not None and z_dict.get("individual"):
            z = z_dict["individual"]["per_head"]
            y = y_dict.get("individual", {})
            if y:
                self.individual_classifier.update_prototypes(z, y)
        if self.group_classifier is not None and z_dict.get("group"):
            z = z_dict["group"]["per_head"]
            y = y_dict.get("group", {})
            if y:
                self.group_classifier.update_prototypes(z, y)

    def _project_branch(self, branch: str, features: torch.Tensor) -> torch.Tensor:
        B, P, D = features.shape
        proj = self.branch_projectors[branch]
        features = proj(features.view(B * P, D)).view(B, P, -1)
        return features

    def _encode_branch(
        self,
        branch: str,
        feat_tensors: Optional[List[torch.Tensor]],
        feat_masks: Optional[List[torch.Tensor]],
        emb_tensor: Optional[torch.Tensor],
        emb_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        spec = BRANCH_MODALITY_MAP[branch]
        encoded: List[torch.Tensor] = []
        sample_shape: Optional[Tuple[int, int]] = None

        if feat_tensors is not None and feat_masks is not None:
            used: set[str] = set()
            for tensor, mask in zip(feat_tensors, feat_masks):
                dim = tensor.shape[-1]
                mod_name = self.feature_dim_to_mod.get(dim)
                if mod_name is None or mod_name not in spec["feat"]:
                    raise ValueError(
                        f"Unrecognised feature dimension {dim} for branch '{branch}'",
                    )
                if mod_name in used:
                    raise ValueError(f"Duplicate modality '{mod_name}' detected in branch '{branch}'")
                used.add(mod_name)
                if sample_shape is None:
                    sample_shape = tensor.shape[:2]
                tensor = tensor.to(device=device, dtype=torch.float32)
                mask_tensor = mask.to(device=device, dtype=torch.float32) if mask is not None else None
                encoded.append(self.sequence_encoders[mod_name](tensor, mask_tensor))

        if emb_tensor is not None:
            emb_name = spec["emb"]
            if sample_shape is None:
                sample_shape = emb_tensor.shape[:2]
            tensor = emb_tensor.to(device=device, dtype=torch.float32)
            mask = emb_mask.to(device=device, dtype=torch.float32) if emb_mask is not None else None
            encoded.append(self.sequence_encoders[emb_name](tensor, mask))

        if not encoded:
            if sample_shape is None:
                raise ValueError(f"No modalities available for branch '{branch}'")
            B, P = sample_shape
            return torch.zeros(B, P, self.hidden_size, device=device)

        stacked = torch.stack(encoded, dim=0).mean(dim=0)
        return self._project_branch(branch, stacked)

    def forward(
        self,
        batch_data: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        epoch: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor | Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        # determine batch shape and device
        first_tensor: Optional[torch.Tensor] = None
        for branch_dict in batch_data.values():
            for mod_name, (mod_tensor, _) in branch_dict.items():
                if mod_tensor is not None:
                    first_tensor = mod_tensor
                    break
            if first_tensor is not None:
                break
        if first_tensor is None:
            raise ValueError("Batch data is empty; cannot infer batch dimensions")
        device = first_tensor.device

        branch_features: Dict[str, torch.Tensor] = {}
        B, P = first_tensor.shape[:2]
        for branch in self.branch_names:
            if branch in batch_data:
                branch_mods = batch_data[branch]
                spec = BRANCH_MODALITY_MAP[branch]
                
                # Extract feature tensors and masks
                feat_tensors: List[torch.Tensor] = []
                feat_masks: List[torch.Tensor] = []
                for mod_name in spec["feat"]:
                    if mod_name in branch_mods:
                        mod_tensor, mod_mask = branch_mods[mod_name]
                        feat_tensors.append(mod_tensor)
                        feat_masks.append(mod_mask)
                
                # Extract embedding tensor and mask
                emb_tensor: Optional[torch.Tensor] = None
                emb_mask: Optional[torch.Tensor] = None
                emb_name = spec["emb"]
                if emb_name in branch_mods:
                    emb_tensor, emb_mask = branch_mods[emb_name]
                
                branch_features[branch] = self._encode_branch(
                    branch,
                    feat_tensors if feat_tensors else None,
                    feat_masks if feat_masks else None,
                    emb_tensor,
                    emb_mask,
                    device,
                )
            else:
                branch_features[branch] = torch.zeros(B, P, self.hidden_size, device=device)

        combined = torch.cat([branch_features[b] for b in self.branch_names], dim=-1)
        combined = self.backbone_norm(combined)
        combined = self.backbone_dropout(combined)

        z_outs: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "backbone": combined,
            "branches": branch_features,
        }
        logits: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.individual_classifier is not None:
            ind_features, ind_logits = self.individual_classifier(combined, epoch)
            z_outs["individual"] = ind_features
            logits["individual"] = ind_logits
        if self.group_classifier is not None:
            grp_features, grp_logits = self.group_classifier(combined, epoch)
            z_outs["group"] = grp_features
            logits["group"] = grp_logits

        return z_outs, logits
