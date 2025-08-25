import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any

def resample_features(
    features_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    resample_freq: int,
    gap_threshold: float = 0.75,
    return_mask: bool = False,
) -> torch.Tensor:
    """
    Resample multivariate time-series features to a uniform time base.

    :param features_df (pd.DataFrame): Input data with a 'timestamp' column and N feature columns.
    :param start_time (float): Start of the target time window.
    :param end_time (float): End of the target time window.
    :param resample_freq (int): Number of samples per second.
    :param gap_threshold (float): Max time delta allowed for interpolation continuity (in seconds).
    :param return_mask (bool): If True, also return a binary mask indicating invalid interpolations.
    :return torch.Tensor: Resampled tensor of shape (num_samples, num_features).
    :return Optional[torch.Tensor]: Binary mask of shape (num_samples,) where True = invalid interpolation.
    """
    if features_df.empty or 'timestamp' not in features_df.columns:
        num_samples = int((end_time - start_time) * resample_freq)
        num_features = len([col for col in features_df.columns if col != 'timestamp'])
        empty_tensor = torch.zeros((num_samples, num_features), dtype=torch.float)
        if return_mask:
            return empty_tensor, torch.zeros_like(empty_tensor, dtype=torch.float)
        return empty_tensor

    # prepare features_df
    features_df = features_df.copy()
    features_df['timestamp'] = pd.to_numeric(features_df['timestamp'], errors='coerce')
    features_df = features_df.sort_values('timestamp').dropna(subset=['timestamp'])

    num_samples = int((end_time - start_time) * resample_freq)
    target_time = np.arange(start_time, end_time, 1.0 / resample_freq)

    # identify chunks separated by large gaps
    features_df['dt'] = features_df['timestamp'].diff().fillna(0)
    features_df['chunk'] = (features_df['dt'] > gap_threshold).cumsum()

    # build mask for valid interpolated points
    grid_df = pd.DataFrame({'timestamp': target_time})
    left = pd.merge_asof(grid_df, features_df[['timestamp', 'chunk']], on='timestamp', direction='backward')
    right = pd.merge_asof(grid_df, features_df[['timestamp', 'chunk']], on='timestamp', direction='forward')
    mask = (left['chunk'] == right['chunk']).values

    # interpolate over the uniform grid
    feature_cols = [col for col in features_df.columns if col not in {'timestamp', 'dt', 'chunk'}]
    resampled = []
    
    for col in feature_cols:
        col_values = pd.to_numeric(features_df[col], errors='coerce')
        interp_values = np.interp(
            target_time,
            features_df['timestamp'],
            col_values,
            left=np.nan,
            right=np.nan,
        )

        # handle NaNs (forward/backward fill if partially missing, else zero)
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

        # mask out invalid interpolations
        interp_values[~mask] = 0.0
        resampled.append(interp_values)

    stacked = np.stack(resampled, axis=1) if resampled else np.zeros((num_samples, 0))
    stacked_tensor = torch.tensor(stacked, dtype=torch.float)

    if return_mask:
        mask_2d = np.tile(mask[:, None], (1, stacked.shape[1])).astype(float)  # shape: (num_samples, num_features)
        return stacked_tensor, torch.tensor(mask_2d, dtype=torch.float)
    return stacked_tensor


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


class StandardizeTransform:
    """Apply group-wise standardization using pre-computed statistics."""

    def __init__(self, stats_dict: Dict[str, Dict[str, Dict[str, float]]]):
        self.stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for mod, groups in stats_dict.items():
            self.stats[mod] = {}
            for grp, ms in groups.items():
                self.stats[mod][grp] = {
                    'mean': torch.as_tensor(ms['mean'], dtype=torch.float),
                    'std': torch.as_tensor(ms['std'], dtype=torch.float),
                }

    def _apply(self, tensor: torch.Tensor, mod: str) -> torch.Tensor:
        if mod not in self.stats:
            return tensor
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
