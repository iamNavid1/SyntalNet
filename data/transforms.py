import numpy as np
import pandas as pd
import torch
from typing import List

def resample_features(
    features_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    num_samples: int
) -> torch.Tensor:
    """
    Resample multivariate time-series features to a uniform time base.

    Args:
        features_df (pd.DataFrame): Input data with a 'timestamp' column and N feature columns.
        start_time (float): Start of the target time window.
        end_time (float): End of the target time window.
        num_samples (int): Number of evenly spaced samples to generate.

    Returns:
        torch.Tensor: Resampled tensor of shape (num_samples, num_features).
    """
    if features_df.empty or 'timestamp' not in features_df.columns:
        return torch.zeros((num_samples, 0), dtype=torch.float)

    # Prepare target time grid
    target_time = np.linspace(start_time, end_time, num=num_samples)
    ts = pd.to_numeric(features_df['timestamp'], errors='coerce')
    features_df = features_df.copy()
    features_df['timestamp'] = ts

    feature_cols = [col for col in features_df.columns if col != 'timestamp']
    resampled = []

    for col in feature_cols:
        x = pd.to_numeric(features_df[col], errors='coerce')
        interp = np.interp(target_time, features_df['timestamp'], x, left=np.nan, right=np.nan)

        # Fill NaNs by forward/backward fill or zero if all NaN
        if np.isnan(interp).all():
            interp[:] = 0.0
        else:
            # Forward fill then back fill
            mask = np.isnan(interp)
            if mask.any():
                valid_idx = ~mask
                interp[mask] = np.interp(
                    target_time[mask],
                    target_time[valid_idx],
                    interp[valid_idx]
                )

        resampled.append(interp)

    stacked = np.stack(resampled, axis=1) if resampled else np.zeros((num_samples, 0))
    return torch.tensor(stacked, dtype=torch.float)









class NormalizeTransform:
    def __init__(self, stats_dict):
        # stats_dict: {modality: {'mean': tensor, 'std': tensor}}
        self.stats = stats_dict

    def __call__(self, sample):
        out = {}
        for mod, data in sample.items():
            mean = self.stats[mod]['mean']
            std = self.stats[mod]['std']
            out[mod] = (data - mean) / std
        return out