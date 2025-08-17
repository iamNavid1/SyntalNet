import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import GroupDynamicsDataset
from collate import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--modalities', nargs='+', default=['face', 'pose', 'video', 'turns', 'utterance', 'sentiment', 'prosody', 'audio'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: root_dir '{args.root_dir}' does not exist or is not a directory.")
        return

    # Instantiate dataset
    prev = time.time()
    dataset = GroupDynamicsDataset(root_dir=args.root_dir, modalities=args.modalities)
    print(f"Dataset with {len(dataset)} samples loaded in {time.time()-prev:.3f}s")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # persistent_workers=True,
        # prefetch_factor=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    prev_time = time.time()
    for batch_idx, (batch_data, batch_labels) in enumerate(loader):
        now = time.time()
        print(f"\nBatch {batch_idx+1} ready in {now-prev_time:.3f}s:", flush=True)
        prev_time = now
        for mod, (data_tensor, mask_tensor) in batch_data.items():
            print(
                f"  Modality '{mod}': data shape {tuple(data_tensor.shape)},",
                f"mask shape {tuple(mask_tensor.shape)}"
            )
        indiv = batch_labels['individual']
        group = batch_labels['group']
        print(f"  Individual labels shape: {tuple(indiv.shape)}")
        print(f"  Group labels shape: {tuple(group.shape)}")

        # if batch_idx >= 2:
        # break


    # # Jump to the 25th batch
    # from itertools import islice
    # batch_tuple = next(islice(loader, 24, None))
    # batch_data, batch_labels = batch_tuple

    # print(f"\nBatch 25:")
    # for mod, (data_tensor, mask_tensor) in batch_data.items():
    #     print(
    #         f"  Modality '{mod}': data shape {tuple(data_tensor.shape)},",
    #         f"mask shape {tuple(mask_tensor.shape)}"
    #     )
    # indiv = batch_labels['individual']
    # group = batch_labels['group']
    # print(f"  Individual labels shape: {tuple(indiv.shape)}")
    # print(f"  Group labels shape: {tuple(group.shape)}")





    # # Find the sample (i, j) with the max valid length in 'turns' and 'utterance' modalities
    # import numpy as np
    # from itertools import islice
    # batch_tuple = next(islice(loader, 24, None))
    # batch_data, batch_labels = batch_tuple
    # # First, find the (batch, person) index with max valid length in 'utterance'
    # utt_data_tensor, utt_mask_tensor = batch_data['utterance']
    # utt_mask_np = utt_mask_tensor.cpu().numpy() if hasattr(utt_mask_tensor, 'cpu') else utt_mask_tensor.numpy()
    # utt_valid_lengths = np.sum(np.all(utt_mask_np, axis=-1), axis=-1)  # shape: [bsz, num_people]
    # max_idx = np.unravel_index(np.argmax(utt_valid_lengths), utt_valid_lengths.shape)
    # max_len_utt = utt_valid_lengths[max_idx]

    # print(f"\nUsing (batch, person) index from 'utterance' with max valid length: {max_idx}, length: {max_len_utt}")

    # for modality in ['turns', 'utterance']:
    #     data_tensor, mask_tensor = batch_data[modality]
    #     mask_np = mask_tensor.cpu().numpy() if hasattr(mask_tensor, 'cpu') else mask_tensor.numpy()
    #     valid_lengths = np.sum(np.all(mask_np, axis=-1), axis=-1)  # shape: [bsz, num_people]
    #     this_len = valid_lengths[max_idx]
    #     data_np = data_tensor.cpu().numpy() if hasattr(data_tensor, 'cpu') else data_tensor.numpy()
    #     person_data = data_np[max_idx[0], max_idx[1], :this_len, :]  # shape [this_len, 8]
    #     print(f"\nModality '{modality}': data for (batch, person) {max_idx} (valid length: {this_len})")
    #     print(f"  Data shape: {person_data.shape}")
    #     for d in person_data:
    #         print(d[:8])
    
    


    
if __name__ == '__main__':
    main()
