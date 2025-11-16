import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from engine.utils import BRANCH_MODALITY_MAP
from models.base_model import BaseModel


FEATURE_DIMS = {
    "face": 212,
    "pose": 182,
    "turns": 12,
    "sentiment": 5,
    "prosody": 7,
}


class _MaskedTemporalLSTM(nn.Module):
    """Simple LSTM encoder for per-person sequences."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm: Optional[nn.LSTM] = None
        self.out_proj: Optional[nn.Sequential] = None

    def _maybe_build(self, input_dim: int, device: torch.device) -> None:
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.lstm.to(device)
        if self.out_proj is None:
            out_dim = self.hidden_size * (2 if self.bidirectional else 1)
            layers: List[nn.Module] = [nn.LayerNorm(out_dim), nn.Linear(out_dim, self.hidden_size), nn.ReLU(inplace=True)]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            self.out_proj = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, P, T, F)
            mask: Optional tensor with same shape as x marking valid entries.
        Returns:
            Tensor of shape (B, P, hidden_size)
        """
        B, P, T, F = x.shape
        device = x.device
        x = x.to(dtype=torch.float32)
        lengths = torch.full((B * P,), T, device=device, dtype=torch.long)
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.float32)
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
                    pad = torch.ones(pad_shape, device=device, dtype=torch.float32)
                    mask = torch.cat([mask, pad], dim=-2)
            mask = mask.clamp_(0.0, 1.0)
            x = x * mask
            mask_lengths = mask.view(B * P, T).sum(dim=-1)
            lengths = mask_lengths.to(dtype=torch.long)
        lengths = lengths.clamp_min(1)

        self._maybe_build(F, device)
        assert self.lstm is not None and self.out_proj is not None

        packed = nn.utils.rnn.pack_padded_sequence(
            x.view(B * P, T, F),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, B * P, self.hidden_size)
            last = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        else:
            last = h_n[-1]
        out = self.out_proj(last)
        out = out.view(B, P, -1)
        return out


class _SimpleClassifier(nn.Module):
    """Lightweight MLP classifier tailored for the LSTM baseline."""

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
            head.K = self.K
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
    def update_prototypes(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op for the LSTM baseline
        return None


class TemporalBiLSTM(BaseModel):
    """Baseline multimodal model with bidirectional LSTM encoders."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        ind_cls_heads: Tuple[str, ...] = ("Engagement", "Lead"),
        grp_cls_heads: Tuple[str, ...] = ("Synchrony", "Confidence", "Transition"),
        classifier_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.branch_names = tuple(BRANCH_MODALITY_MAP.keys())
        self.feature_dim_to_mod = {FEATURE_DIMS[name]: name for name in FEATURE_DIMS}

        modality_names = sorted(
            {
                mod
                for spec in BRANCH_MODALITY_MAP.values()
                for mod in [*spec["feat"], spec["emb"]]
                if mod is not None
            }
        )
        self.sequence_encoders = nn.ModuleDict(
            {
                mod: _MaskedTemporalLSTM(
                    hidden_size=hidden_size,
                    num_layers=num_lstm_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                for mod in modality_names
            }
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
        batch_data: Dict[str, Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]],
        epoch: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor | Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        first_tensor: Optional[torch.Tensor] = None
        for values in batch_data.values():
            feat_list, _, emb_tensor, _ = values
            if feat_list:
                first_tensor = feat_list[0]
                break
            if emb_tensor is not None:
                first_tensor = emb_tensor
                break
        if first_tensor is None:
            raise ValueError("Batch data is empty; cannot infer batch dimensions")
        device = first_tensor.device

        branch_features: Dict[str, torch.Tensor] = {}
        B, P = first_tensor.shape[:2]
        for branch in self.branch_names:
            if branch in batch_data:
                feat_tensors, feat_masks, emb_tensor, emb_mask = batch_data[branch]
                branch_features[branch] = self._encode_branch(
                    branch,
                    feat_tensors,
                    feat_masks,
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