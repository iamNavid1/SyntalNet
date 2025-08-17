import os
import glob
import random
import copy
import json
import re
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import resample_features

COLUMNS_TO_KEEP = {
    'face': [
        "gaze_angle_x", "gaze_angle_y",
        "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz",
        *[f"{coord}_{i}" for coord in ['X', 'Y', 'Z'] for i in range(68)]
    ],
    'turns': [
        'Turn Position', 'Turn Duration', 'Pause Before',
        'Speaker Change?', 'Has Overlap?', 'Is Floor-taking?',
        'Is Butting-in?', 'Is Backchannel?'
    ],
    'prosody': [
        'pitch_Hz','hnr_dB', 'mfcc_1', 'energy_dB', 
        'jitter_percent', 'shimmer_dB', 'percent_silence'
    ],
    'individual_labels': [
        ['Q4_L_label', 'Q5_L_label'],
        ['Q4_M_label', 'Q5_M_label'], 
        ['Q4_R_label', 'Q5_R_label']
    ],
    'group_labels': [
        'Q1_label', 'Q2_label', 'Q3_label',
    ]
}


class GroupDynamicsDataset(Dataset):
    """
    Multimodal dataset for human group dynamics.
    Generates sliding-window samples per-person per-group.
    """
    def __init__(
        self,
        root_dir,
        modalities,
        snippet_length = 10,    # sec
        stride = 3,          # sec
        resample_freq = 10,  # hz
        transforms=None
    ):
        self.root_dir = root_dir
        self.modalities = modalities
        self.snippet_length = snippet_length
        self.stride = stride
        self.len_overlap = snippet_length % stride
        self.resample_freq = resample_freq
        self.transforms = transforms

        # File-level caches
        self._csv_cache: Dict[str, pd.DataFrame] = {}
        self._json_cache: Dict[str, Any] = {}
        self._npy_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Scan for group IDs
        example_mod = modalities[0]
        pattern_csv = os.path.join(root_dir, example_mod, "Group_*.csv")
        pattern_json = os.path.join(root_dir, example_mod, "Group_*.json")
        files = glob.glob(pattern_csv) + glob.glob(pattern_json)
        group_ids = set()
        for f in files:
            base = os.path.basename(f)
            match = re.match(r"Group_(\d+)\.(csv|json)$", base)
            if match:
                group_ids.add(int(match.group(1)))
        self.group_ids = sorted(group_ids)

        # Prepare samples: list of dicts
        self.samples = []  # each entry: { 'group': int, 'start_time': float }
        for gid in self.group_ids:
            person_timestamps = {}  # per-person timestamps
            csv_path = os.path.join(root_dir, example_mod, f"Group_{gid:02}.csv")
            json_path = os.path.join(root_dir, example_mod, f"Group_{gid:02}.json")

            if os.path.exists(csv_path):
                df = self._load_csv(csv_path)
                for pid in range(3):
                    person_timestamps[pid] = df[df['face_id'] == pid+1]['timestamp'].values

            elif os.path.exists(json_path):
                frames = self._load_json(json_path)
                for pid in range(3):
                    person_timestamps[pid] = np.array([
                        frame[pid]['timestamp'] for frame in frames
                    ])

            else:
                raise FileNotFoundError(f"No CSV or JSON found for group {gid} in modality {example_mod}")

            t0 = max([ts[0] for ts in person_timestamps.values() if len(ts) > 0])
            tN = min([ts[-1] for ts in person_timestamps.values() if len(ts) > 0])

            start = int(np.ceil(t0 / self.stride)) * self.stride
            while start + 2*self.snippet_length - self.len_overlap <= tN:
                self.samples.append({'group': gid, 'start_time': start})
                start += self.stride

    def __len__(self):
        return len(self.samples)

    def _load_csv(self, path: str) -> pd.DataFrame:
        if path not in self._csv_cache:
            self._csv_cache[path] = pd.read_csv(path)
        return self._csv_cache[path]

    def _load_json(self, path: str) -> Any:
        if path not in self._json_cache:
            with open(path, 'r') as f:
                self._json_cache[path] = json.load(f)
        return self._json_cache[path]

    def _load_npy(self, path: str) -> np.memmap:
        if path in self._npy_cache:
            self._npy_cache.move_to_end(path)
        else:
            self._npy_cache[path] = np.load(path)
            if len(self._npy_cache) > 1000:
                self._npy_cache.popitem(last=False)
        return self._npy_cache[path]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        gid = sample['group']
        t0 = sample['start_time']
        t1 = t0 + self.snippet_length * 2 - self.len_overlap

        modality_data = {}
        modality_mask = {}

        for mod in self.modalities:
            folder = os.path.join(self.root_dir, mod)

            if mod.lower() == 'face':
                # csv-based face loader
                csv_path = os.path.join(folder, f"Group_{gid:02}.csv")
                df = self._load_csv(csv_path)

                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    mask = (
                        (df['timestamp'] >= t0) &
                        (df['timestamp'] <= t1) &
                        (df['face_id'] == pid)
                    )
                    person_df = df.loc[mask, ['timestamp'] + COLUMNS_TO_KEEP[mod]]
                    feat_tensor, mask_tensor = resample_features(person_df, t0, t1, self.resample_freq, return_mask=True)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, num_frames, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif mod.lower() == 'pose':
                # json-based pose loader
                json_path = os.path.join(folder, f"Group_{gid:02}.json")
                frames = self._load_json(json_path)

                num_joints = 32
                all_feat = []
                all_mask = []
                for pid in range(3):
                    records = []
                    for frame in frames:
                        person = frame[pid]
                        ts = person['timestamp']
                        if not (t0 <= ts <= t1):
                            continue
                        pos = person['joint_positions']
                        ori = person['joint_orientations']
                        if pos and ori:
                            flat_pos = {
                                f"pos_{j}_{axis}": pos[j][k]
                                for j in range(num_joints)
                                for axis, k in [('x', 0), ('y', 1), ('z', 2)]
                            }
                            flat_ori = {
                                f"ori_{j}_{quat}": ori[j][k]
                                for j in range(num_joints)
                                for quat, k in zip(['w', 'x', 'y', 'z'], range(4))
                            }
                            flat = {**flat_pos, **flat_ori}
                        else:
                            flat = {
                                f"pos_{j}_{axis}": np.nan
                                for j in range(num_joints)
                                for axis in ('x', 'y', 'z')
                            }
                            flat.update({
                                f"ori_{j}_{quat}": np.nan
                                for j in range(num_joints)
                                for quat in ('w', 'x', 'y', 'z')
                            })
                        records.append({'timestamp': ts, **flat})

                    df_person = pd.DataFrame.from_records(records)
                    feat_tensor, mask_tensor = resample_features(df_person, t0, t1, self.resample_freq, return_mask=True)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, num_frames, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif mod.lower() == 'video':
                # npy-based embedding loadedr (pre-computed for clips of 19 sec)
                clip_i = t0 // self.stride

                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    arr = self._load_npy(npy_i_path)
                    feat_tensor = torch.tensor(arr, dtype=torch.float)
                    mask_tensor = torch.ones_like(feat_tensor, dtype=torch.bool)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, 57, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif mod.lower() == 'turns':
                # csv-based turns loader with intervals
                csv_path = os.path.join(folder, f"Group_{gid:02}.csv")
                df = self._load_csv(csv_path)
                mask_overlap = ((df['Start'] <= t1) & (df['End'] >= t0))

                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    mask_spk = df[f"Speaker {pid}"] == 1
                    sub = df.loc[mask_overlap & mask_spk,
                                 COLUMNS_TO_KEEP['turns']]
                    sub = sub.replace({'True': 1, 'False': 0})
                    sub = sub.astype(float)
                    feat_tensor = torch.tensor(sub.values, dtype=float)
                    mask_tensor = torch.ones_like(feat_tensor, dtype=torch.bool)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, num_turns, ...) *variable length sequence
                modality_data[mod] = all_feat
                modality_mask[mod] = all_mask

            elif mod.lower() == 'utterance':
                # npy-based embedding loadedr (pre-computed for clips of 19 sec)
                clip_i = t0 // self.stride

                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    arr = self._load_npy(npy_i_path)
                    feat_tensor = torch.tensor(arr, dtype=torch.float)
                    mask_tensor = torch.ones_like(feat_tensor, dtype=torch.bool)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)                    
                # shape: (num_person, num_turns, ...) *variable length sequence
                modality_data[mod] = all_feat
                modality_mask[mod] = all_mask

            elif mod.lower() == 'audio':
                # npy-based data loader for precomputed audio embedding sampled at 10 HZ with 1-sec sliding window
                # merge two consecuive 10-sec clips
                clip_i = t0 // self.stride
                clip_j = clip_i + (self.snippet_length // self.stride)
                prev_anchor = clip_i - (self.snippet_length // self.stride)
                next_anchor = clip_j + (self.snippet_length // self.stride)
                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    npy_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.npy")
                    arrs = [self._load_npy(path) for path in (npy_i_path, npy_j_path)]
                    merged = np.concatenate([arrs[0], arrs[1][1:]], axis=0)  # shape: (181, 512)
                    # retrieve the first 5 and last 4 embs from anchors
                    anchor_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{prev_anchor+1}.npy")
                    anchor_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{next_anchor+1}.npy")
                    anchors = []
                    for path in (anchor_i_path, anchor_j_path):
                        if os.path.exists(path):
                            anchors.append(self._load_npy(path))
                        else:
                            anchors.append(None)
                    if anchors[0] is not None:
                        merged = np.concatenate([anchors[0][-6:-1], merged], axis=0)
                    else:
                        merged = np.concatenate([np.repeat(merged[0:1], 5, axis=0), merged], axis=0)
                    if anchors[1] is not None:
                        merged = np.concatenate([merged, anchors[1][1:5]])
                    else:
                        merged = np.concatenate([merged, np.repeat(merged[-1:], 4, axis=0)], axis=0)
                    feat_tensor = torch.tensor(merged, dtype=torch.float)
                    mask_tensor = torch.ones_like(feat_tensor, dtype=torch.bool)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, (2*clip_length-overlap)*resample_freq, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif mod.lower() == 'sentiment':
                # npy-based data loader for audio sentiment sampled at 10 HZ with 1-sec sliding window
                # merge two consecuive 10-sec clips
                clip_i = t0 // self.stride
                clip_j = clip_i + (self.snippet_length // self.stride)
                prev_anchor = clip_i - (self.snippet_length // self.stride)
                next_anchor = clip_j + (self.snippet_length // self.stride)
                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    npy_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.npy")
                    arrs = [self._load_npy(path) for path in (npy_i_path, npy_j_path)]
                    merged = np.concatenate([arrs[0], arrs[1][1:]], axis=0)  # shape: (181, 5)
                    # retrieve the first 5 and last 4 embs from anchors
                    anchor_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{prev_anchor+1}.npy")
                    anchor_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{next_anchor+1}.npy")
                    anchors = []
                    for path in (anchor_i_path, anchor_j_path):
                        if os.path.exists(path):
                            anchors.append(self._load_npy(path))
                        else:
                            anchors.append(None)
                    if anchors[0] is not None:
                        merged = np.concatenate([anchors[0][-6:-1], merged], axis=0)
                    else:
                        merged = np.concatenate([np.repeat(merged[0:1], 5, axis=0), merged], axis=0)
                    if anchors[1] is not None:
                        merged = np.concatenate([merged, anchors[1][1:5]])
                    else:
                        merged = np.concatenate([merged, np.repeat(merged[-1:], 4, axis=0)], axis=0)
                    feat_tensor = torch.tensor(merged, dtype=torch.float)
                    mask_tensor = torch.ones_like(feat_tensor, dtype=torch.bool)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, (2*clip_length-overlap)*resample_freq, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif mod.lower() == 'prosody':
                # csv-based data loader for prosodic and spectral features with variable sampling rate
                # merge two consecuive 10-sec clips
                clip_i = t0 // self.stride
                clip_j = clip_i + (self.snippet_length // self.stride)
                prev_anchor = clip_i - (self.snippet_length // self.stride)
                next_anchor = clip_j + (self.snippet_length // self.stride)
                all_feat = []
                all_mask = []
                for pid in range(1, 4):
                    csv_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.csv")
                    csv_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.csv")
                    df1 = self._load_csv(csv_i_path)
                    df2 = self._load_csv(csv_j_path)
                    df1['timestamp'] = pd.to_numeric(df1['timestamp']) + clip_i * self.stride  # absolute timestamps
                    df2['timestamp'] = pd.to_numeric(df2['timestamp']) + clip_j * self.stride  # absolute timestamps
                    pre_overlap_mask = df1['timestamp'] < t0 + self.snippet_length - self.len_overlap
                    post_overlap_mask = df2['timestamp'] > t0 + self.len_overlap
                    df1_pre = df1.loc[pre_overlap_mask, COLUMNS_TO_KEEP[mod]]
                    df2_post = df2.loc[post_overlap_mask, COLUMNS_TO_KEEP[mod]]
                    overlap_time = np.arange(t0 + self.snippet_length - self.len_overlap, t0 + self.snippet_length + 1e-8, 1/10)
                    overlap_cols = {}
                    for col in df1.columns:
                        if col in COLUMNS_TO_KEEP[mod]:
                            interp1 = np.interp(overlap_time, df1['timestamp'], df1[col], left=np.nan, right=np.nan)
                            interp2 = np.interp(overlap_time, df2['timestamp'], df2[col], left=np.nan, right=np.nan)
                            valid1 = ~np.isnan(interp1)
                            valid2 = ~np.isnan(interp2)
                            avg = np.where(valid1 & valid2, (interp1 + interp2) / 2,
                                        np.where(valid1, interp1,
                                                    np.where(valid2, interp2, np.nan)))
                            overlap_cols[col] = avg
                    df_overlap = pd.DataFrame({'timestamp': overlap_time, **overlap_cols})
                    merged = pd.concat([df1_pre, df_overlap, df2_post], ignore_index=True)
                    feat_tensor, mask_tensor = resample_features(merged, t0, t1, self.resample_freq, return_mask=True)
                    all_feat.append(feat_tensor)
                    all_mask.append(mask_tensor)
                # shape: (num_person, (2*clip_length-overlap)*resample_freq, ...)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            else:
                raise ValueError(f"Unknown modality: {mod}")

        # Load and process labels for individual and group data
        csv_path = os.path.join(self.root_dir, f"labels/annotation_summary_majority_vote.csv")
        labels_df = self._load_csv(csv_path)

        clip_i = t0 // self.stride
        clip_j = clip_i + (self.snippet_length // self.stride)
        video_name_i = f"Group{gid:02}_Clip{clip_i+1}"
        video_name_j = f"Group{gid:02}_Clip{clip_j+1}"

        row_i = labels_df[labels_df['Video'] == video_name_i]
        row_j = labels_df[labels_df['Video'] == video_name_j]

        labels = {}

        # Individual label evolution (Q4, Q5)
        individual_lables = np.full((3, 2), -1, dtype=np.int64)  # default -1 for missing
        if not row_i.empty and not row_j.empty:
            for p_idx, person in enumerate(COLUMNS_TO_KEEP['individual_labels']):
                for l_idx, label in enumerate(person):
                    value_i = row_i[label].iloc[0]
                    value_j = row_j[label].iloc[0]
                    if value_j < value_i:
                        individual_lables[p_idx, l_idx] = 0  # Decreasing
                    elif value_j == value_i:
                        individual_lables[p_idx, l_idx] = 1  # Staying the same
                    else:  # value_j > value_i
                        individual_lables[p_idx, l_idx] = 2  # Increasing
        # else: already filled with -1

        # Group label evolution (Q1, Q2, Q3)
        group_labels = np.full(3, -1, dtype=np.int64)
        if not row_i.empty and not row_j.empty:
            for l_idx, label in enumerate(COLUMNS_TO_KEEP['group_labels']):
                value_i = row_i[label].iloc[0]
                value_j = row_j[label].iloc[0]
                if value_j < value_i:
                    group_labels[l_idx] = 0  # Decreasing
                elif value_j == value_i:
                    group_labels[l_idx] = 1  # Staying the same
                else:
                    group_labels[l_idx] = 2  # Increasing
        # else: already filled with -1

        labels['individual'] = torch.tensor(individual_lables, dtype=torch.long)
        labels['group'] = torch.tensor(group_labels, dtype=torch.long)

        if self.transforms:
            modality_data = self.transforms(modality_data)

        return modality_data, modality_mask, labels, gid



