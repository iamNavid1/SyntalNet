import os
import glob
import random
import copy
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from transforms import resample_features

COLUMNS_TO_KEEP = {
    'face': [
        "gaze_angle_x", "gaze_angle_y",
        "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz",
        *[f"{coord}_{i}" for coord in ['x', 'y', 'X', 'Y', 'Z'] for i in range(68)]
    ],
    'turns': [
        'Turn Position', 'Turn Duration', 'Pause Before',
        'Speaker Change?', 'Has Overlap?', 'Is Floor-taking?',
        'Is Butting-in?', 'Is Backchannel?'
    ],
    'audio': [
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
        window_length,
        stride,
        transforms=None
    ):
        self.root_dir = root_dir
        self.modalities = modalities
        self.window_length = window_length
        self.stride = stride
        self.transforms = transforms

        # Collect group IDs by scanning one modality
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
                df = pd.read_csv(csv_path)
                for pid in range(3):
                    person_timestamps[pid] = df[df['face_id'] == pid+1]['timestamp'].values

            elif os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    frames = json.load(f)
                for pid in range(3):
                    person_timestamps[pid] = np.array([
                        frame[pid]['timestamp'] for frame in frames
                    ])

            else:
                raise FileNotFoundError(f"No CSV or JSON found for group {gid} in modality {example_mod}")

            t0 = max([ts[0] for ts in person_timestamps.values() if len(ts) > 0])
            tN = min([ts[-1] for ts in person_timestamps.values() if len(ts) > 0])

            start = ((int(np.ceil(t0 / self.stride))) * self.stride)
            while start + self.window_length <= tN:
                self.samples.append({'group': gid, 'start_time': start})
                start += self.stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        gid = sample['group']
        t0 = sample['start_time']
        t1 = t0 + self.window_length

        modality_data = {}

        for mod in self.modalities:
            folder = os.path.join(self.root_dir, mod)

            if mod.lower() == 'face':
                # csv-based face loader
                csv_path = os.path.join(folder, f"Group_{gid:02}.csv")
                df = pd.read_csv(csv_path)

                people_data = []
                for pid in range(1, 4):
                    mask = (
                        (df['timestamp'] >= t0) &
                        (df['timestamp'] < t1) &
                        (df['face_id'] == pid)
                    )
                    person_df = df.loc[mask, COLUMNS_TO_KEEP[mod]]
                    people_data.append(torch.tensor(person_df.values, dtype=torch.float))
                # shape: (num_person, num_frames, ...)
                modality_data[mod] = torch.stack(people_data, dim=0)

            elif mod.lower() == 'pose':
                # json-based pose loader
                json_path = os.path.join(folder, f"Group_{gid:02}.json")
                with open(json_path, 'r') as f:
                    frames = json.load(f)

                people_data = []
                for pid in range(3):
                    timestamps = [frame[pid]['timestamp'] for frame in frames]
                    frame_idxs = [i for i, ts in enumerate(timestamps) if t0 <= ts < t1]
                    # extract pose: shape (num_frames, 32, 3)
                    poses = [np.array(frames[i][pid]['joint_positions']) for i in frame_idxs]
                    people_data.append(torch.tensor(np.stack(poses), dtype=torch.float))
                # shape: (num_person, num_frames, ...)
                modality_data[mod] = torch.stack(people_data, dim=0)

            elif mod.lower() == 'turns':
                # csv-based turns loader with intervals
                csv_path = os.path.join(folder, f"Group_{gid:02}.csv")
                df = pd.read_csv(csv_path)
                mask_overlap = (df['Start'] <= t1 &
                                df['End'] >= t0)

                people_data = []
                for pid in range(1, 4):
                    mask_spk = df[f"Speaker {pid}"] == 1
                    sub = df.loc[mask_overlap & mask_spk,
                                 COLUMNS_TO_KEEP['turns']]
                    people_data.append(torch.tensor(sub.values, dtype=float))
                # shape: (num_person, num_turns, ...) *variable length sequence
                modality_data[mod] = people_data

            elif mod.lower() in ['utterance', 'video']:
                # npy-based embedding loadedr (pre-computed)
                # merge two consecuive 10-sec clips
                clip_i = t0 // self.stride
                clip_j = clip_i + (10 // self.stride)

                def _deduplicate(clip_i, clip_j, atol=1e-6):  # only for utterance
                    if clip_i.size == 0:
                        return clip_j
                    if clip_j.size == 0:
                        return clip_i
                    max_overlap = min(len(clip_i), len(clip_j))
                    overlap_len = 0
                    for k in range(max_overlap, 0, -1):
                        suffix_i = clip_i[-k:]           # last k rows of clip_i
                        prefix_j = clip_j[:k]            # first k rows of clip_j
                        if np.allclose(suffix_i, prefix_j, atol=atol):
                            overlap_len = k
                            break
                    merged = np.concatenate([clip_i, clip_j[overlap_len:]], axis=0)
                    return merged

                people_data = []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    npy_j_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.npy")
                    arrs = [np.load(path) for path in (npy_i_path, npy_j_path)]
                    if mod == 'utterance':
                        merged = _deduplicate(arrs[0], arrs[1])
                    else:
                        merged = np.concatenate(arrs, axis=0)
                    people_data.append(torch.tensor(merged, dtype=torch.float))
                # shape: 
                #       utterance -> (num_person, num_turns, ...) *variable length sequence
                #       video     -> (num_person, 20, ...)
                modality_data[mod] = people_data

            elif mod.lower() == 'audio':
                # both csv and npy-based data loader for embedding, sentiment, feature
                # merge two consecuive 10-sec clips
                clip_i = t0 // self.stride
                clip_j = clip_i + (10 // self.stride)
                prev_anchor = clip_i - (10 // self.stride)
                next_anchor = clip_j + (10 // self.stride)
                people_data = []
                for pid in range(1, 4):
                    # Embeddings (sampled at 1 sec sliding windows with 0.1 sec stride)
                    npy_i_path = os.path.join(folder, 'embedding', f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    npy_j_path = os.path.join(folder, 'embedding', f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.npy")
                    arrs = [np.load(path) for path in (npy_i_path, npy_j_path)]
                    merged_emb = np.concatenate([arrs[0], arrs[1][1:]], axis=0)  # shape: (181, 512)
                    # retrieve the first 5 and last 4 embs from anchors
                    anchor_i_path = os.path.join(folder, 'embedding', f"Group{gid:02}_Person{pid}_Clip{prev_anchor+1}.npy")
                    anchor_j_path = os.path.join(folder, 'embedding', f"Group{gid:02}_Person{pid}_Clip{next_anchor+1}.npy")
                    anchors = []
                    for path in (anchor_i_path, anchor_j_path):
                        if os.path.exists(path):
                            anchors.append(np.load(path))
                        else:
                            anchors.append(None)
                    if anchors[0]:
                        merged_emb = np.concatenate([anchors[0][-6:-1], merged_emb], axis=0)
                    else:
                        merged_emb = np.concatenate([np.repeat(merged_emb[0], 5, axis=0), merged_emb], axis=0)
                    if anchors[1]:
                        merged_emb = np.concatenate([merged_emb, anchors[1][1:5]])
                    else:
                        merged_emb = np.concatenate([merged_emb, np.repeat(merged_emb[-1], 4, axis=0)], axis=0)
                    merged_emb = torch.tensor(merged_emb, dtype=torch.float)

                    # Sentiment (sampled at 10 HZ)
                    npy_i_path = os.path.join(folder, 'sentiment', f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    npy_j_path = os.path.join(folder, 'sentiment', f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.npy")
                    arrs = [np.load(path) for path in (npy_i_path, npy_j_path)]
                    merged_sent = np.concatenate([arrs[0], arrs[1][1:]], axis=0)  # shape: (181, 512)
                    # retrieve the first 5 and last 4 embs from anchors
                    anchor_i_path = os.path.join(folder, 'sentiment', f"Group{gid:02}_Person{pid}_Clip{prev_anchor+1}.npy")
                    anchor_j_path = os.path.join(folder, 'sentiment', f"Group{gid:02}_Person{pid}_Clip{next_anchor+1}.npy")
                    anchors = []
                    for path in (anchor_i_path, anchor_j_path):
                        if os.path.exists(path):
                            anchors.append(np.load(path))
                        else:
                            anchors.append(None)
                    if anchors[0]:
                        merged_sent = np.concatenate([anchors[0][-6:-1], merged_sent], axis=0)
                    else:
                        merged_sent = np.concatenate([np.repeat(merged_sent[0], 5, axis=0), merged_sent], axis=0)
                    if anchors[1]:
                        merged_sent = np.concatenate([merged_sent, anchors[1][1:5]])
                    else:
                        merged_sent = np.concatenate([merged_sent, np.repeat(merged_sent[-1], 4, axis=0)], axis=0)
                    merged_sent = torch.tensor(merged_sent, dtype=torch.float)

                    # Feature (variable sampling rate)
                    csv_i_path = os.path.join(folder, 'feature', f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.csv")
                    csv_j_path = os.path.join(folder, 'feature', f"Group{gid:02}_Person{pid}_Clip{clip_j+1}.csv")
                    df1 = pd.read_csv(csv_i_path)
                    df2 = pd.read_csv(csv_j_path)
                    df1['timestamp'] = pd.to_numeric(df1['timestamp']) + clip_i * self.stride  # absolute timestamps
                    df2['timestamp'] = pd.to_numeric(df2['timestamp']) + clip_j * self.stride  # absolute timestamps
                    len_overlap = 10 % self.stride
                    pre_overlap_mask = df1['timestamp'] < t0 + 10 - len_overlap
                    post_overlap_mask = df2['timestamp'] > t1 - 10 + len_overlap
                    df1_pre = df1.loc[pre_overlap_mask, COLUMNS_TO_KEEP[mod]]
                    df2_post = df2.loc[post_overlap_mask, COLUMNS_TO_KEEP[mod]]
                    overlap_time = np.arange(t0 + 10 - len_overlap, t1 - 10 + len_overlap + 1e-8, 1/10)
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
                    merged_df = pd.concat([df1_pre, df_overlap, df2_post], ignore_index=True)
                    merged_feat = resample_features(merged_df, t0, t1, )

                    people_data.append(torch.cat([merged_emb, merged_sent, merged_feat], dim=1))
                # shape: (num_person, 10*window_length, ...)
                modality_data[mod] = torch.stack(people_data, dim=0)

            else:
                raise ValueError(f"Unknown modality: {mod}")

        # Load and process labels for individual and group data
        csv_path = os.path.join(self.root_dir, f"labels/annotation_summary_majority_vote.csv")
        labels_df = pd.read_csv(csv_path)

        clip_i = t0 // self.stride
        clip_j = clip_i + (10 // self.stride)
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

        return modality_data, labels, gid



