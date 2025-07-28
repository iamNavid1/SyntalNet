import torch

def accuracy(preds, labels):
    _, preds_max = torch.max(preds, 1)
    correct = (preds_max == labels).sum().item()
    return correct / labels.size(0)

# More metrics (precision, recall, F1, AUC) must be added