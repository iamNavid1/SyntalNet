import torch


def _pad_across_batch(data_list, mask_list, pad_value=0):
    """
    Pads a list of per-sample tensors to the maximum sequence length across batch.

    :param data_list: List[Tensor], each of shape (num_person, L_i, feat_dim)
    :param mask_list: List[BoolTensor], same shapes as data_list
    :param pad_value: value to use for padding the data tensor
    :returns padded_data: Tensor of shape (batch, num_person, L_max, feat_dim)
    :returns padded_mask: BoolTensor of same shape
    """
    batch_size = len(data_list)
    num_person, _, feat_dim = data_list[0].shape
    L_max = max(x.shape[1] for x in data_list)

    padded_data = torch.full((batch_size, num_person, L_max, feat_dim), pad_value, dtype=data_list[0].dtype)
    padded_mask = torch.zeros((batch_size, num_person, L_max, feat_dim), dtype=torch.bool)

    for i, (x, m) in enumerate(zip(data_list, mask_list)):
        L = x.shape[1]
        padded_data[i, :, :L, :] = x
        padded_mask[i, :, :L, :] = m

    return padded_data, padded_mask


def _pad_within_sample(data_list, mask_list, pad_value=0):
    """
    Pads per-person sequences within a single sample to the maximum length among persons.

    :param data_list: List[Tensor], each of shape (L_j, feat_dim)
    :param mask_list: List[BoolTensor], same shapes
    :param pad_value: value to use for padding the data tensor
    :returns data_tensor: Tensor of shape (num_person, L_sample_max, feat_dim)
    :returns mask_tensor: BoolTensor of same shape
    """
    num_person = len(data_list)
    feat_dim = data_list[0].shape[1]
    L_sample_max = max(x.shape[0] for x in data_list)

    data_tensor = torch.full((num_person, L_sample_max, feat_dim), pad_value, dtype=data_list[0].dtype)
    mask_tensor = torch.zeros((num_person, L_sample_max, feat_dim), dtype=torch.bool)

    for j, (x, m) in enumerate(zip(data_list, mask_list)):
        L = x.shape[0]
        data_tensor[j, :L, :] = x
        mask_tensor[j, :L, :] = m

    return data_tensor, mask_tensor


def collate_fn(batch):
    """
    Collate function for GroupDynamicsDataset.

    :param batch: list of tuples (modality_data, modality_mask, labels, gid)
    :returns batch_data: dict mapping each modality to a tuple (data_tensor, mask_tensor)
    :returns batch_labels: dict with 'individual' and 'group' label tensors
    """
    batch_data = {}
    batch_labels = {}

    modalities = batch[0][0].keys()

    for mod in modalities:
        data_items = [sample[0][mod] for sample in batch]
        mask_items = [sample[1][mod] for sample in batch]

        # handle per-person variable-length sequences
        if isinstance(data_items[0], list):
            # pad within each sample to unify person sequence lengths
            per_sample_data = []
            per_sample_mask = []
            for data_list, mask_list in zip(data_items, mask_items):
                sample_data, sample_mask = _pad_within_sample(data_list, mask_list)
                per_sample_data.append(sample_data)
                per_sample_mask.append(sample_mask)
            # pad across batch to unify sequence lengths
            padded_data, padded_mask = _pad_across_batch(per_sample_data, per_sample_mask)
            batch_data[mod] = (padded_data, padded_mask)
        else:
            # fixed-size tensors: stack directly
            x = torch.stack(data_items, dim=0)
            m = torch.stack(mask_items, dim=0)
            batch_data[mod] = (x, m)

    batch_labels['individual'] = torch.stack([sample[2]['individual'] for sample in batch], dim=0)
    batch_labels['group']      = torch.stack([sample[2]['group']     for sample in batch], dim=0)

    return batch_data, batch_labels
