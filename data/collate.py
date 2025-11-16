from __future__ import annotations
import torch

def _pad_across_batch(data_list, mask_list, pad_value=0.0):
    """
    Pads a list of per-sample tensors to the max sequence length across batch.
    Expects data: (P, L_i, F), mask: (P, L_i, 1) as float32 (1.0 = valid).
    Returns data: (B, P, L_max, F), mask: (B, P, L_max, 1)
    """
    batch_size = len(data_list)
    num_person, _, feat_dim = data_list[0].shape
    L_max = max(max(x.shape[1] for x in data_list), 5)

    padded_data = torch.full(
        (batch_size, num_person, L_max, feat_dim),
        float(pad_value),
        dtype=data_list[0].dtype,
    )
    padded_mask = torch.zeros((batch_size, num_person, L_max, 1), dtype=torch.float32)

    for i, (x, m) in enumerate(zip(data_list, mask_list)):
        L = x.shape[1]
        padded_data[i, :, :L, :] = x
        padded_mask[i, :, :L, :] = m

    return padded_data, padded_mask


def _pad_within_sample(data_list, mask_list, pad_value=0.0):
    """
    Pads per-person variable-length sequences within a single sample to the
    max length among persons.
    data_list: List[(L_j, F)]
    mask_list: List[(L_j, 1)]
    Returns: (P, L_max, F), (P, L_max, 1)
    """
    num_person = len(data_list)
    feat_dim = data_list[0].shape[1]
    L_sample_max = max(x.shape[0] for x in data_list)

    data_tensor = torch.full(
        (num_person, L_sample_max, feat_dim),
        float(pad_value),
        dtype=data_list[0].dtype,
    )
    mask_tensor = torch.zeros((num_person, L_sample_max, 1), dtype=torch.float32)

    for j, (x, m) in enumerate(zip(data_list, mask_list)):
        L = x.shape[0]
        data_tensor[j, :L, :] = x
        mask_tensor[j, :L, :] = m

    return data_tensor, mask_tensor


def collate_fn(batch):
    """
    Collate function for GroupDynamicsDataset.

    :param batch: list of tuples (modality_data, modality_mask, labels, gid)
    :returns batch_data: dict mapping modality -> (data_tensor, mask_tensor)
    :returns batch_labels: dict with 'individual' and 'group'
    """
    batch_data = {}
    batch_labels = {}

    modalities = batch[0][0].keys()

    for mod in modalities:
        data_items = [sample[0][mod] for sample in batch]
        mask_items = [sample[1][mod] for sample in batch]

        # Variable-length per-person list (e.g., turns, utterance)
        if isinstance(data_items[0], list):
            per_sample_data, per_sample_mask = [], []
            for data_list, mask_list in zip(data_items, mask_items):
                sample_data, sample_mask = _pad_within_sample(data_list, mask_list)
                per_sample_data.append(sample_data)
                per_sample_mask.append(sample_mask)

            padded_data, padded_mask = _pad_across_batch(per_sample_data, per_sample_mask)
            batch_data[mod] = (padded_data, padded_mask)

        else:
            # Fixed-size tensors -> stack directly
            x = torch.stack(data_items, dim=0)  # (B,P,L,F)
            m = torch.stack(mask_items, dim=0)  # (B,P,L,1)
            # Ensure float mask
            if m.dtype != torch.float32:
                m = m.to(torch.float32)
            batch_data[mod] = (x, m)

    batch_labels['individual'] = torch.stack([sample[2]['individual'] for sample in batch], dim=0)
    batch_labels['group']      = torch.stack([sample[2]['group']      for sample in batch], dim=0)

    return batch_data, batch_labels
