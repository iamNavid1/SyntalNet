import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple, Any


def resample_features(
    features_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    resample_freq: int,
    gap_threshold: float = 0.75,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Resample multivariate time-series features to a uniform grid.
    - Interpolates each column independently over target_time.
    - Builds a continuity mask by chunking gaps > gap_threshold.
    Returns:
      features: (L, F) float32
      mask:     (L, 1) float32 (1.0 valid), if return_mask
    """
    # Compute grid
    num_samples = int((end_time - start_time) * resample_freq)
    target_time = np.arange(start_time, end_time, 1.0 / resample_freq)
    if target_time.shape[0] != num_samples:
        # guard against floating point drift
        target_time = np.linspace(start_time, end_time, num_samples, endpoint=False)

    # Handle degenerate input quickly
    if features_df is None or features_df.empty or 'timestamp' not in features_df.columns:
        num_features = len([col for col in features_df.columns if col != 'timestamp'])
        empty = torch.zeros((num_samples, num_features), dtype=torch.float32)
        if return_mask:
            return empty, torch.zeros((num_samples, 1), dtype=torch.float32)
        return empty

    # Clean/sort timestamps
    df = features_df.copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').dropna(subset=['timestamp'])

    # Identify continuity chunks based on gaps
    df['dt'] = df['timestamp'].diff().fillna(0)
    df['chunk'] = (df['dt'] > gap_threshold).cumsum()

    feature_cols = [c for c in df.columns if c not in {'timestamp', 'dt', 'chunk'}]

    # Build continuity mask on the grid using forward/backward chunk ids
    grid_df = pd.DataFrame({'timestamp': target_time})
    left = pd.merge_asof(grid_df, df[['timestamp', 'chunk']], on='timestamp', direction='backward')
    right = pd.merge_asof(grid_df, df[['timestamp', 'chunk']], on='timestamp', direction='forward')
    cont_mask = (left['chunk'].values == right['chunk'].values)

    # Interpolate each feature over the grid
    resampled = []
    for col in feature_cols:
        col_values = pd.to_numeric(df[col], errors='coerce')
        interp_values = np.interp(
            target_time,
            df['timestamp'].values,
            col_values.values,
            left=np.nan,
            right=np.nan,
        )
        # If all NaN, set zeros; else ffill/bfill within the target grid
        if np.isnan(interp_values).all():
            interp_values[:] = 0.0
        else:
            nan_mask = np.isnan(interp_values)
            if nan_mask.any():
                valid_idx = ~nan_mask
                interp_values[nan_mask] = np.interp(
                    target_time[nan_mask],
                    target_time[valid_idx],
                    interp_values[valid_idx],
                )
        # Zero-out invalid interpolations across gaps
        interp_values[~cont_mask] = 0.0
        resampled.append(interp_values)

    if resampled:
        stacked = np.stack(resampled, axis=1)  # (L, F)
    else:
        stacked = np.zeros((num_samples, 0), dtype=np.float32)

    feats = torch.tensor(stacked, dtype=torch.float32)

    if return_mask:
        # Return time mask as (L,1) float32
        mask_1d = torch.from_numpy(cont_mask.astype(np.float32)).unsqueeze(1)  # (L,1)
        return feats, mask_1d
    return feats


STANDARDIZATION_GROUPS: Dict[str, Dict[str, List[int]]] = {
    'face': {
        'gaze_angle_x': [0],
        'gaze_angle_y': [1],
        'pose_Tx': [2],
        'pose_Ty': [3],
        'pose_Tz': [4],
        'pose_Rx': [5],
        'pose_Ry': [6],
        'pose_Rz': [7],
        'coords_X': [8 + 3 * i for i in range(68)],
        'coords_Y': [8 + 3 * i + 1 for i in range(68)],
        'coords_Z': [8 + 3 * i + 2 for i in range(68)],
    },
    'pose': {
        'joint_pos_x': [3 * i for i in range(26)],
        'joint_pos_y': [3 * i + 1 for i in range(26)],
        'joint_pos_z': [3 * i + 2 for i in range(26)],
        'joint_ori_w': [78 + 4 * i for i in range(26)],
        'joint_ori_x': [78 + 4 * i + 1 for i in range(26)],
        'joint_ori_y': [78 + 4 * i + 2 for i in range(26)],
        'joint_ori_z': [78 + 4 * i + 3 for i in range(26)],
    },
    'turns': {
        'turn_position': [0],
        'turn_duration': [1],
        'cumulative_turn_duration': [2],
        'pause_before': [3],
        'binary_flags': [4, 5, 6, 7, 8],
        'cumulative_floor_taking': [9],
        'cumulative_butting_in': [10],
        'cumulative_backchannel': [11],
    },
    'sentiment': {
        'logits': list(range(5)),
    },
    'prosody': {
        'pitch_Hz': [0],
        'hnr_dB': [1],
        'mfcc_1': [2],
        'energy_dB': [3],
        'jitter_percent': [4],
        'shimmer_dB': [5],
        'percent_silence': [6],
    },
}

EMBEDDING_DIMS: Dict[str, int] = {
    'video': 1024,
    'utterance': 1024,
    'audio': 512,
}


class StandardizeTransform:
    """Apply group-wise standardization using pre-computed statistics."""

    def __init__(self, stats_dict: Dict[str, Dict[str, Dict[str, float]]]):
        self.stats: Dict[str, Any] = {}
        for mod, groups in stats_dict.items():
            if mod in EMBEDDING_DIMS:
                self.stats[mod] = {
                    'mean': torch.as_tensor(groups['mean'], dtype=torch.float),
                    'std': torch.as_tensor(groups['std'], dtype=torch.float),
                }
            else:
                self.stats[mod] = {}
                for grp, ms in groups.items():
                    self.stats[mod][grp] = {
                        'mean': torch.as_tensor(ms['mean'], dtype=torch.float),
                        'std': torch.as_tensor(ms['std'], dtype=torch.float),
                    }

    def _apply(self, tensor: torch.Tensor, mod: str) -> torch.Tensor:
        if mod not in self.stats:
            return tensor

        if mod in EMBEDDING_DIMS:
            mean = self.stats[mod]['mean'].to(tensor.device)
            std = self.stats[mod]['std'].to(tensor.device)
            return (tensor - mean) / (std + 1e-6)

        out = tensor.clone()
        for grp, idxs in STANDARDIZATION_GROUPS.get(mod, {}).items():
            if grp not in self.stats[mod]:
                continue
            mean = self.stats[mod][grp]['mean'].to(tensor.device)
            std = self.stats[mod][grp]['std'].to(tensor.device)
            out[..., idxs] = (out[..., idxs] - mean) / (std + 1e-6)
        return out

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for mod, data in sample.items():
            if isinstance(data, list):
                out[mod] = [self._apply(d, mod) for d in data]
            else:
                out[mod] = self._apply(data, mod)
        return out
