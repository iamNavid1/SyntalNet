from __future__ import annotations

import os
import glob
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import resample_features


COLUMNS_TO_KEEP = {
    'face': [
        "gaze_angle_x", "gaze_angle_y",
        "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz",
        *[f"{coord}_{i}" for coord in ['X', 'Y', 'Z'] for i in range(68)]
    ],
    'turns': [
        'Turn Position', 'Turn Duration', 'Cumulative Turn Duration', 'Pause Before',
        'Speaker Change?', 'Has Overlap?', 'Is Floor-taking?', 'Is Butting-in?', 'Is Backchannel?',
        'Cumulative Floor-taking', 'Cumulative Butting-in', 'Cumulative Backchannel'
    ],
    'prosody': [
        'pitch_Hz', 'hnr_dB', 'mfcc_1', 'energy_dB',
        'jitter_percent', 'shimmer_dB', 'percent_silence'
    ],
    'individual_labels': [
        ['Q4_L_{type}', 'Q5_L_{type}'],
        ['Q4_M_{type}', 'Q5_M_{type}'],
        ['Q4_R_{type}', 'Q5_R_{type}']
    ],
    'group_labels': [
        'Q1_{type}', 'Q2_{type}', 'Q3_{type}',
    ]
}


class GroupDynamicsDataset(Dataset):
    """
    Multimodal dataset for human group dynamics.
    Generates sliding-window samples per-person per-group.
    """
    def __init__(
        self,
        root_dir:       str,
        modalities:     List[str],
        snippet_length: int                 = 10,        # sec
        stride:         int                 = 3,         # sec
        resample_freq:  int                 = 10,        # hz
        label_type:     str                 = 'kernel',  # label | ema | kernel | kalman
        transforms:     Optional[Any]       = None,
        include_groups: Optional[List[int]] = None,
        exclude_groups: Optional[List[int]] = None,
        # Tunables for caches/behavior:
        npy_cache_cap:  int                 = 256,
        audio_merge_cache_cap: int          = 2048,
    ):
        self.root_dir = root_dir
        self.modalities = modalities
        self.snippet_length = snippet_length
        self.stride = stride
        self.len_overlap = snippet_length % stride
        self.resample_freq = resample_freq
        self.label_type = label_type
        self.transforms = transforms

        # ---------- low-level caches ----------
        self._csv_cache: Dict[str, pd.DataFrame] = {}
        self._json_cache: Dict[str, Any] = {}
        self._npy_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._npy_cache_cap = int(npy_cache_cap)

        # ---------- higher-level caches ----------
        self._face_by_gid: Dict[int, List[pd.DataFrame]] = {}
        self._pose_by_gid: Dict[int, List[pd.DataFrame]] = {}
        self._turns_by_gid: Dict[int, pd.DataFrame] = {}
        self._prosody_clip_cache: Dict[Tuple[int, int, int], pd.DataFrame] = {}
        # merged audio/sentiment cache: (mod,gid,pid,clip_i,clip_j,prev_anchor,next_anchor) -> np.ndarray
        self._audio_merge_cache: OrderedDict[
            Tuple[str,int,int,int,int,int,int], np.ndarray
        ] = OrderedDict()
        self._audio_merge_cache_cap = int(audio_merge_cache_cap)

        # ---------- group discovery ----------
        pattern = os.path.join(root_dir, "face", "Group_*.csv")
        files = glob.glob(pattern)
        group_ids = set()
        for f in files:
            base = os.path.basename(f)
            m = re.match(r"Group_(\d+)\.(csv|json)$", base)
            if m:
                group_ids.add(int(m.group(1)))
        self.group_ids = sorted(group_ids)

        if include_groups is not None:
            self.group_ids = [g for g in self.group_ids if g in include_groups]
        if exclude_groups is not None:
            self.group_ids = [g for g in self.group_ids if g not in exclude_groups]

        # ---------- labels map ----------
        labels_csv = os.path.join(self.root_dir, 'labels', 'annotation_summary.csv')
        labels_df = self._load_csv(labels_csv)
        self._label_map: Dict[str, Dict[str, Any]] = labels_df.set_index('Video').to_dict('index')

        # ---------- sample index ----------
        self.samples: List[Dict[str, int]] = []
        for gid in self.group_ids:
            face_csv = os.path.join(root_dir, "face", f"Group_{gid:02}.csv")
            df = self._load_csv(face_csv)
            self._ensure_face_split(gid, df)

            person_ts: Dict[int, np.ndarray] = {}
            for pid in range(3):
                pdf = self._face_by_gid[gid][pid]
                person_ts[pid] = pdf['timestamp'].to_numpy()

            # Require overlap across all 3 people
            t0 = max([ts[0] for ts in person_ts.values() if ts.size > 0])
            tN = min([ts[-1] for ts in person_ts.values() if ts.size > 0])

            start = int(np.ceil(t0 / self.stride)) * self.stride
            while start + 2 * self.snippet_length - self.len_overlap <= tN:
                self.samples.append({'group': gid, 'start_time': start})
                start += self.stride

    # ---------- cached I/O ----------
    def _load_csv(self, path: str) -> pd.DataFrame:
        if path not in self._csv_cache:
            self._csv_cache[path] = pd.read_csv(path)
        return self._csv_cache[path]

    def _load_json(self, path: str) -> Any:
        if path not in self._json_cache:
            with open(path, 'r') as f:
                self._json_cache[path] = json.load(f)
        return self._json_cache[path]

    def _close_memmap(self, arr: np.ndarray):
        mmap_obj = getattr(arr, '_mmap', None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except ValueError:
                pass

    def _clear_npy_cache(self):
        while self._npy_cache:
            _, arr = self._npy_cache.popitem(last=False)
            self._close_memmap(arr)

    def _load_npy(self, path: str) -> np.ndarray:
        # memmap so OS shares pages across workers
        if path in self._npy_cache:
            self._npy_cache.move_to_end(path)
        else:
            self._npy_cache[path] = np.load(path, mmap_mode='r')
            if len(self._npy_cache) > self._npy_cache_cap:
                _, evicted = self._npy_cache.popitem(last=False)
                self._close_memmap(evicted)
        return self._npy_cache[path]

    # ---------- higher-level prep ----------
    def _ensure_face_split(self, gid: int, df_face: Optional[pd.DataFrame] = None):
        if gid in self._face_by_gid:
            return
        if df_face is None:
            df_face = self._load_csv(os.path.join(self.root_dir, 'face', f'Group_{gid:02}.csv'))
        per = []
        cols = ['timestamp'] + COLUMNS_TO_KEEP['face']
        for pid in range(1, 4):
            pdf = df_face.loc[df_face['face_id'] == pid, cols]
            per.append(pdf.reset_index(drop=True))
        self._face_by_gid[gid] = per

    def _ensure_pose_group(self, gid: int):
        if gid in self._pose_by_gid:
            return
        json_path = os.path.join(self.root_dir, 'pose', f'Group_{gid:02}.json')
        frames = self._load_json(json_path)

        num_joints = 26
        pose_cols = {f"pos_{j}_{axis}": np.nan for j in range(num_joints) for axis in ('x','y','z')}
        ori_cols  = {f"ori_{j}_{q}":   np.nan for j in range(num_joints) for q in ('w','x','y','z')}
        nan_template = {**pose_cols, **ori_cols}

        per = [[] for _ in range(3)]
        for frame in frames:
            for pid in range(3):
                person = frame[pid]
                row = {'timestamp': person['timestamp']}
                pos = person['joint_positions']
                ori = person['joint_orientations']
                if pos and ori:
                    for j in range(num_joints):
                        x, y, z = pos[j]
                        w, ox, oy, oz = ori[j]
                        row[f"pos_{j}_x"] = x; row[f"pos_{j}_y"] = y; row[f"pos_{j}_z"] = z
                        row[f"ori_{j}_w"] = w; row[f"ori_{j}_x"] = ox; row[f"ori_{j}_y"] = oy; row[f"ori_{j}_z"] = oz
                else:
                    row.update(nan_template)
                per[pid].append(row)
        self._pose_by_gid[gid] = [pd.DataFrame(p) for p in per]

    def _get_prosody_clip(self, gid: int, pid: int, clip_idx: int) -> pd.DataFrame:
        key = (gid, pid, clip_idx)
        if key in self._prosody_clip_cache:
            return self._prosody_clip_cache[key]
        folder = os.path.join(self.root_dir, 'prosody')
        csv_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_idx+1}.csv")
        df = self._load_csv(csv_path).copy()
        # make timestamps absolute so we can concatenate clips quickly
        df['timestamp'] = pd.to_numeric(df['timestamp']) + clip_idx * self.stride
        self._prosody_clip_cache[key] = df
        return df

    # ---------- audio/sentiment merge cache ----------
    def _get_merged_audio_like(
        self,
        mod: str,
        gid: int,
        pid: int,
        clip_i: int,
        clip_j: int,
        prev_anchor: int,
        next_anchor: int,
    ) -> np.ndarray:
        """
        Returns merged (anchors + clip_i + clip_j) array for audio-like modalities.
        Cached per (mod,gid,pid,clip_i,clip_j,prev_anchor,next_anchor).
        """
        key = (mod, gid, pid, clip_i, clip_j, prev_anchor, next_anchor)
        if key in self._audio_merge_cache:
            self._audio_merge_cache.move_to_end(key)
            return self._audio_merge_cache[key]

        folder = os.path.join(self.root_dir, mod)
        base = f"Group{gid:02}_Person{pid}_Clip"

        arr_i = self._load_npy(os.path.join(folder, f"{base}{clip_i+1}.npy"))
        arr_j = self._load_npy(os.path.join(folder, f"{base}{clip_j+1}.npy"))
        merged = np.concatenate([arr_i, arr_j[1:]], axis=0)

        # anchors (previous and next)
        ai_path = os.path.join(folder, f"{base}{prev_anchor+1}.npy")
        aj_path = os.path.join(folder, f"{base}{next_anchor+1}.npy")

        if os.path.exists(ai_path):
            ai = self._load_npy(ai_path)
            merged = np.concatenate([ai[-6:-1], merged], axis=0)
        else:
            merged = np.concatenate([np.repeat(merged[0:1], 5, axis=0), merged], axis=0)

        if os.path.exists(aj_path):
            aj = self._load_npy(aj_path)
            merged = np.concatenate([merged, aj[1:5]], axis=0)
        else:
            merged = np.concatenate([merged, np.repeat(merged[-1:], 4, axis=0)], axis=0)

        # LRU update
        self._audio_merge_cache[key] = merged
        if len(self._audio_merge_cache) > self._audio_merge_cache_cap:
            self._audio_merge_cache.popitem(last=False)
        return merged

    # ---------- utils ----------
    @staticmethod
    def _to_time_mask(mask_like: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Ensure mask is (L,1) float32 (1.0 = valid).
        If a feature-dim mask is provided, collapse via any() then cast to float.
        """
        if mask_like.ndim == 2 and mask_like.shape[0] == feat.shape[0]:
            if mask_like.shape[1] == 1:
                return mask_like.to(torch.float32)
            return (mask_like != 0).any(dim=1, keepdim=True).to(torch.float32)
        return torch.ones((feat.shape[0], 1), dtype=torch.float32)

    @staticmethod
    def _np_to_torch_f32(arr: np.ndarray) -> torch.Tensor:
        if not getattr(arr, 'flags', None) or not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        return torch.as_tensor(arr, dtype=torch.float32)

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        gid = sample['group']
        t0 = sample['start_time']
        t1 = t0 + self.snippet_length * 2 - self.len_overlap

        modality_data: Dict[str, Any] = {}
        modality_mask: Dict[str, Any] = {}

        for mod in self.modalities:
            folder = os.path.join(self.root_dir, mod)
            m = mod.lower()

            if m == 'face':
                self._ensure_face_split(gid)
                all_feat, all_mask = [], []
                cols = ['timestamp'] + COLUMNS_TO_KEEP['face']
                for pid in range(3):
                    pdf = self._face_by_gid[gid][pid]
                    mask = (pdf['timestamp'] >= t0) & (pdf['timestamp'] <= t1)
                    person_df = pdf.loc[mask, cols]
                    feat, mask_t = resample_features(
                        person_df, t0, t1, self.resample_freq,
                        gap_threshold=0.75,
                        return_mask=True
                    )
                    feat = feat.contiguous()
                    mask_t = self._to_time_mask(mask_t, feat)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = torch.stack(all_feat, dim=0)       # (P,L,F)
                modality_mask[mod] = torch.stack(all_mask, dim=0)       # (P,L,1)

            elif m == 'pose':
                self._ensure_pose_group(gid)
                all_feat, all_mask = [], []
                for pid in range(3):
                    dfp = self._pose_by_gid[gid][pid]
                    sub = dfp[(dfp['timestamp'] >= t0) & (dfp['timestamp'] <= t1)]
                    feat, mask_t = resample_features(
                        sub, t0, t1, self.resample_freq,
                        gap_threshold=0.75,
                        return_mask=True
                    )
                    feat = feat.contiguous()
                    mask_t = self._to_time_mask(mask_t, feat)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif m == 'video':
                clip_i = t0 // self.stride
                all_feat, all_mask = [], []
                for pid in range(1, 4):
                    npy_i_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    arr = self._load_npy(npy_i_path)
                    feat = self._np_to_torch_f32(arr).to(torch.float32)
                    mask_t = torch.ones((feat.shape[0], 1), dtype=torch.float32)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif m == 'turns':
                if gid not in self._turns_by_gid:
                    csv_path = os.path.join(folder, f"Group_{gid:02}.csv")
                    df = self._load_csv(csv_path).copy()
                    df[COLUMNS_TO_KEEP['turns']] = df[COLUMNS_TO_KEEP['turns']].replace({'True': 1, 'False': 0}).astype(float)
                    self._turns_by_gid[gid] = df
                df = self._turns_by_gid[gid]
                mask_overlap = ((df['Start'] <= t1) & (df['End'] >= t0))

                all_feat, all_mask = [], []
                cols = COLUMNS_TO_KEEP['turns']
                for pid in range(1, 4):
                    mask_spk = df[f"Speaker {pid}"] == 1
                    sub = df.loc[mask_overlap & mask_spk, cols]
                    feat = self._np_to_torch_f32(sub.to_numpy(copy=False))
                    mask_t = torch.ones((feat.shape[0], 1), dtype=torch.float32)
                    all_feat.append(feat)   # list of (T,F)
                    all_mask.append(mask_t) # list of (T,1)
                modality_data[mod] = all_feat      # per-person lists (variable length)
                modality_mask[mod] = all_mask

            elif m == 'utterance':
                clip_i = t0 // self.stride
                all_feat, all_mask = [], []
                for pid in range(1, 4):
                    npy_path = os.path.join(folder, f"Group{gid:02}_Person{pid}_Clip{clip_i+1}.npy")
                    arr = self._load_npy(npy_path)
                    feat = self._np_to_torch_f32(arr)
                    mask_t = torch.ones((feat.shape[0], 1), dtype=torch.float32)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = all_feat
                modality_mask[mod] = all_mask

            elif m in {'audio', 'sentiment'}:
                clip_i = t0 // self.stride
                clip_j = clip_i + (self.snippet_length // self.stride)
                prev_anchor = clip_i - (self.snippet_length // self.stride)
                next_anchor = clip_j + (self.snippet_length // self.stride)

                all_feat, all_mask = [], []
                for pid in range(1, 4):
                    merged = self._get_merged_audio_like(
                        m, gid, pid, clip_i, clip_j, prev_anchor, next_anchor
                    )
                    feat = self._np_to_torch_f32(merged)
                    mask_t = torch.ones((feat.shape[0], 1), dtype=torch.float32)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            elif m == 'prosody':
                clip_i = t0 // self.stride
                clip_j = clip_i + (self.snippet_length // self.stride)
                all_feat, all_mask = [], []
                for pid in range(1, 4):
                    df1 = self._get_prosody_clip(gid, pid, clip_i)
                    df2 = self._get_prosody_clip(gid, pid, clip_j)

                    pre_overlap_mask = df1['timestamp'] < t0 + self.snippet_length - self.len_overlap
                    post_overlap_mask = df2['timestamp'] > t0 + self.len_overlap
                    df1_pre = df1.loc[pre_overlap_mask, ['timestamp'] + COLUMNS_TO_KEEP['prosody']]
                    df2_post = df2.loc[post_overlap_mask, ['timestamp'] + COLUMNS_TO_KEEP['prosody']]

                    step = 1/self.resample_freq
                    overlap_time = np.arange(
                        t0 + self.snippet_length - self.len_overlap,
                        t0 + self.snippet_length + 1e-8,
                        step
                    )
                    overlap_cols = {}
                    for col in COLUMNS_TO_KEEP['prosody']:
                        v1 = np.interp(overlap_time, df1['timestamp'].to_numpy(), df1[col].to_numpy(), left=np.nan, right=np.nan)
                        v2 = np.interp(overlap_time, df2['timestamp'].to_numpy(), df2[col].to_numpy(), left=np.nan, right=np.nan)
                        valid1, valid2 = ~np.isnan(v1), ~np.isnan(v2)
                        avg = np.where(valid1 & valid2, (v1 + v2) / 2,
                                       np.where(valid1, v1, np.where(valid2, v2, np.nan)))
                        overlap_cols[col] = avg
                    df_overlap = pd.DataFrame({'timestamp': overlap_time, **overlap_cols})
                    merged = pd.concat([df1_pre, df_overlap, df2_post], ignore_index=True)

                    feat, mask_t = resample_features(
                        merged, t0, t1, self.resample_freq,
                        gap_threshold=0.75,
                        return_mask=True
                    )
                    feat = feat.contiguous()
                    mask_t = self._to_time_mask(mask_t, feat)
                    all_feat.append(feat)
                    all_mask.append(mask_t)
                modality_data[mod] = torch.stack(all_feat, dim=0)
                modality_mask[mod] = torch.stack(all_mask, dim=0)

            else:
                raise ValueError(f"Unknown modality: {mod}")

        # ---------- labels ----------
        clip_i = t0 // self.stride
        clip_j = clip_i + (self.snippet_length // self.stride)
        video_name_i = f"Group{gid:02}_Clip{clip_i+1}"
        video_name_j = f"Group{gid:02}_Clip{clip_j+1}"

        ri = self._label_map.get(video_name_i)
        rj = self._label_map.get(video_name_j)

        labels: Dict[str, torch.Tensor] = {}
        individual = np.full((3, 2), -1, dtype=np.int64)
        group = np.full(3, -1, dtype=np.int64)
        if ri is None or rj is None:
            print(f"Missing label: G{gid:02} {video_name_i} or {video_name_j}")
        else:
            # individual
            for p_idx, person in enumerate(COLUMNS_TO_KEEP['individual_labels']):
                for l_idx, lbl in enumerate(person):
                    col = lbl.format(type=self.label_type)
                    vi = ri[col]; vj = rj[col]
                    individual[p_idx, l_idx] = 0 if vj < vi else (1 if vj == vi else 2)
            # group
            for l_idx, lbl in enumerate(COLUMNS_TO_KEEP['group_labels']):
                col = lbl.format(type=self.label_type)
                vi = ri[col]; vj = rj[col]
                group[l_idx] = 0 if vj < vi else (1 if vj == vi else 2)

        labels['individual'] = torch.from_numpy(individual)
        labels['group']      = torch.from_numpy(group)

        if self.transforms:
            modality_data = self.transforms(modality_data)

        return modality_data, modality_mask, labels, gid

    def close(self):
        """Release file-backed caches to free file descriptors."""
        self._clear_npy_cache()
        self._audio_merge_cache.clear()
        self._prosody_clip_cache.clear()
        self._csv_cache.clear()
        self._json_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
