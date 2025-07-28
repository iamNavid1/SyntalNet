import torch
from torch.utils.data import DataLoader
from data.dataset import GroupDynamicsDataset

# Example parameters (adjust as needed for your data structure)
root_dir = '/home/npargoo/Desktop/wtd/scripts/main/dataset/processed'
modalities = ['pose']
window_length = 10
stride = 3

# Instantiate the dataset
try:
    dataset = GroupDynamicsDataset(
        root_dir=root_dir,
        modalities=modalities,
        window_length=window_length,
        stride=stride
    )
except Exception as e:
    print(f"Failed to instantiate dataset: {e}")
    exit(1)

print(f"Dataset length: {len(dataset)}")

# Create a DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through a few batches
for i, (data, labels, gid) in enumerate(loader):
    print(f"Batch {i}")
    print("Group IDs:", gid)
    for mod, tensor in data.items():
        print(f"  Modality: {mod}, type: {type(tensor)}, shape: {getattr(tensor, 'shape', None)}")
    print("Labels:", labels)
    if i >= 2:
        break 