import numpy as np
import pandas as pd
import torch
from typing import List

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

    Args:
        features_df (pd.DataFrame): Input data with a 'timestamp' column and N feature columns.
        start_time (float): Start of the target time window.
        end_time (float): End of the target time window.
        resample_freq (int): Number of samples per second.
        gap_threshold (float): Max time delta allowed for interpolation continuity (in seconds).
        return_mask (bool): If True, also return a boolean mask indicating invalid interpolations.

    Returns:
        torch.Tensor: Resampled tensor of shape (num_samples, num_features).
        Optional[torch.Tensor]: Boolean mask of shape (num_samples,) where True = invalid interpolation.
    """
    if features_df.empty or 'timestamp' not in features_df.columns:
        num_samples = int((end_time - start_time) * resample_freq)
        num_features = len([col for col in features_df.columns if col != 'timestamp'])
        empty_tensor = torch.zeros((num_samples, num_features), dtype=torch.float)
        return (empty_tensor, torch.zeros_like(empty_tensor, dtype=torch.bool)) if return_mask else empty_tensor

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
            right=np.nan
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
                    interp_values[valid_idx]
                )

        # mask out invalid interpolations
        interp_values[~mask] = 0.0
        resampled.append(interp_values)

    stacked = np.stack(resampled, axis=1) if resampled else np.zeros((num_samples, 0))
    stacked_tensor = torch.tensor(stacked, dtype=torch.float)

    if return_mask:
        mask_2d = np.tile(mask[:, None], (1, stacked.shape[1])).astype(bool)  # shape: (num_samples, num_features)
        return stacked_tensor, torch.tensor(mask_2d, dtype=torch.bool)
    else:
        return stacked_tensor





# class NormalizeTransform:
#     def __init__(self, stats_dict):
#         # stats_dict: {modality: {'mean': tensor, 'std': tensor}}
#         self.stats = stats_dict

#     def __call__(self, sample):
#         out = {}
#         for mod, data in sample.items():
#             mean = self.stats[mod]['mean']
#             std = self.stats[mod]['std']
#             out[mod] = (data - mean) / std
#         return out