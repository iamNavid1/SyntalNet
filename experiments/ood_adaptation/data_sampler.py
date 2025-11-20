from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, List


class StratifiedSampler:
    """
    Sample a subset of data with stratification based on labels.
    Ensures representation of all classes in the sampled subset.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def sample_indices(
        self, 
        dataset: Dataset,
        proportion: float,
        label_key: str = "individual"
    ) -> Tuple[List[int], List[int]]:
        """
        Sample a proportion of the dataset with stratification.
        
        Args:
            dataset: The dataset to sample from
            proportion: Proportion of data to sample (0.0 to 1.0)
            label_key: Which label to use for stratification ("individual" or "group")
        
        Returns:
            Tuple of (sampled_indices, remaining_indices)
        """
        n_total = len(dataset)
        
        if proportion == 0.0:
            return [], list(range(n_total))
        
        if proportion >= 1.0:
            return list(range(n_total)), []
        
        # Collect all labels for stratification
        all_labels = []
        for i in range(n_total):
            _, labels = dataset[i]
            # Get first label dimension for stratification
            if label_key in labels:
                label = labels[label_key]
                # For individual labels: (P, L), take first person's first label
                # For group labels: (L,), take first label
                if label.ndim == 2:
                    label_val = label[0, 0].item()
                else:
                    label_val = label[0].item()
                all_labels.append(label_val)
            else:
                # Fallback: no stratification
                all_labels.append(0)
        
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        
        sampled_indices = []
        
        # Sample from each class proportionally
        for label in unique_labels:
            class_indices = np.where(all_labels == label)[0]
            n_class = len(class_indices)
            n_sample = max(1, int(np.round(n_class * proportion)))
            
            # Ensure we don't oversample
            n_sample = min(n_sample, n_class)
            
            # Sample without replacement
            sampled = self.rng.choice(class_indices, size=n_sample, replace=False)
            sampled_indices.extend(sampled.tolist())
        
        # Get remaining indices
        all_indices = set(range(n_total))
        sampled_set = set(sampled_indices)
        remaining_indices = sorted(list(all_indices - sampled_set))
        
        return sampled_indices, remaining_indices


def create_finetune_test_split(
    dataset: Dataset,
    finetune_proportion: float,
    seed: int = 42,
    stratify: bool = True,
    label_key: str = "individual"
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into fine-tuning and test subsets.
    
    Args:
        dataset: The full dataset (typically the left-out group for a fold)
        finetune_proportion: Proportion of data for fine-tuning (0.0 to 1.0)
        seed: Random seed for reproducibility
        stratify: Whether to use stratified sampling
        label_key: Which label to use for stratification
    
    Returns:
        Tuple of (finetune_subset, test_subset)
    """
    if stratify:
        sampler = StratifiedSampler(seed=seed)
        finetune_indices, test_indices = sampler.sample_indices(
            dataset, finetune_proportion, label_key
        )
    else:
        # Simple random sampling
        n_total = len(dataset)
        n_finetune = int(np.round(n_total * finetune_proportion))
        
        rng = np.random.RandomState(seed)
        all_indices = np.arange(n_total)
        rng.shuffle(all_indices)
        
        finetune_indices = all_indices[:n_finetune].tolist()
        test_indices = all_indices[n_finetune:].tolist()
    
    finetune_subset = Subset(dataset, finetune_indices)
    test_subset = Subset(dataset, test_indices)
    
    return finetune_subset, test_subset


def create_zero_shot_test(dataset: Dataset) -> Subset:
    """
    Create a test set using all data (for 0% fine-tuning baseline).
    
    Args:
        dataset: The full dataset
    
    Returns:
        Test subset (all data)
    """
    all_indices = list(range(len(dataset)))
    return Subset(dataset, all_indices)

