import sys
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

VALID_LABELS = {0, 1, 2}


def majority_label(values, rng=None):
    """
    Compute the majority label among VALID_LABELS.
    - Filters strictly to {0,1,2}.
    - Returns None if no valid labels.
    - Tie-breaking:
        * If 1 ties with any non-1, choose randomly among the non-1 ties.
        * Otherwise choose randomly among the tied modes.
    """
    rng = rng or random
    vals = [int(v) for v in values if int(v) in VALID_LABELS]
    if not vals:
        return None
    c = Counter(vals)
    top = max(c.values())
    modes = [k for k, v in c.items() if v == top]

    if len(modes) == 1:
        return modes[0]

    # Prefer non-1 if 1 is in the tie
    if 1 in modes:
        non_one = [m for m in modes if m != 1]
        if non_one:
            return rng.choice(non_one)

    # Otherwise random among tied modes
    return rng.choice(modes)


def has_any_0_or_2(values):
    """
    Presence check on raw labels (strict {0,1,2}):
    - Returns True if any valid label is 0 or 2.
    - Returns False if valid labels exist and all are 1.
    - Returns None if no valid labels exist.
    """
    vals = [int(v) for v in values if int(v) in VALID_LABELS]
    if not vals:
        return None
    return any(v in (0, 2) for v in vals)


def to_percent_distribution(counter):
    """Return {0: %, 1: %, 2: %} over the counts of valid labels; fills missing keys with 0.0."""
    total = sum(counter.get(k, 0) for k in VALID_LABELS)
    if total == 0:
        return {k: 0.0 for k in VALID_LABELS}
    return {k: counter.get(k, 0) * 100.0 / total for k in VALID_LABELS}


def to_percent_presence(d):
    """
    d = {'0_or_2': count, 'only_1': count}
    Return percentages over these two keys; 0.0 if total == 0.
    """
    total = sum(d.values())
    if total == 0:
        return {'0_or_2': 0.0, 'only_1': 0.0}
    return {k: v * 100.0 / total for k, v in d.items()}


def compute_kernel_label_statistics(root_dir='./dataset', seed=0):
    """
    Computes:
      - Majority distributions for group (3 labels), individual (6 labels), and all (9 labels).
      - Presence stats (any 0/2 vs only 1) on raw labels for group, individual, all.
    """
    # Optional: set seed for reproducibility of tie-breaking
    rng = random.Random(seed) if seed is not None else random

    from data.dataset import GroupDynamicsDataset

    ds = GroupDynamicsDataset(
        root_dir=str(root_dir),
        modalities=['face'],
        label_type='kernel',
    )

    maj_group = Counter()
    maj_ind = Counter()
    maj_all = Counter()

    presence = {
        'group': {'0_or_2': 0, 'only_1': 0},
        'individual': {'0_or_2': 0, 'only_1': 0},
        'all': {'0_or_2': 0, 'only_1': 0},
    }

    for i in range(len(ds)):
        _, _, labels, _ = ds[i]

        # --- Collect raw valid labels ---
        # Group: 3 labels
        group_vals = []
        for v in getattr(labels['group'], 'tolist', lambda: labels['group'])():
            v = int(v)
            if v in VALID_LABELS:
                group_vals.append(v)

        # Individual: 3 people × 2 heads => flatten to 6 labels
        ind_flat = []
        ind_src = getattr(labels['individual'], 'tolist', lambda: labels['individual'])()
        # Ensure we can iterate rows -> cols (list of lists or similar)
        for row in ind_src:
            for v in (row if isinstance(row, (list, tuple)) else [row]):
                v = int(v)
                if v in VALID_LABELS:
                    ind_flat.append(v)

        all_vals = group_vals + ind_flat

        # --- Majority over raw labels with tie preferring non-1 ---
        mg = majority_label(group_vals, rng) if group_vals else None
        mi = majority_label(ind_flat, rng) if ind_flat else None
        ma = majority_label(all_vals, rng) if all_vals else None

        if mg is not None:
            maj_group[mg] += 1
        if mi is not None:
            maj_ind[mi] += 1
        if ma is not None:
            maj_all[ma] += 1

        # --- Presence stats (on raw labels, not majorities) ---
        pg = has_any_0_or_2(group_vals)
        if pg is True:
            presence['group']['0_or_2'] += 1
        elif pg is False:
            presence['group']['only_1'] += 1

        pi = has_any_0_or_2(ind_flat)
        if pi is True:
            presence['individual']['0_or_2'] += 1
        elif pi is False:
            presence['individual']['only_1'] += 1

        pa = has_any_0_or_2(all_vals)
        if pa is True:
            presence['all']['0_or_2'] += 1
        elif pa is False:
            presence['all']['only_1'] += 1

    # --- Summaries ---
    results = {
        'majority': {
            'group': to_percent_distribution(maj_group),
            'individual': to_percent_distribution(maj_ind),
            'all': to_percent_distribution(maj_all),
        },
        'presence': {
            'group': to_percent_presence(presence['group']),
            'individual': to_percent_presence(presence['individual']),
            'all': to_percent_presence(presence['all']),
        },
        'raw_counts': {
            'majority': {'group': maj_group, 'individual': maj_ind, 'all': maj_all},
            'presence': presence,
        }
    }
    return results


def pretty_print(results):
    print('Single label distribution (majorities over raw labels):')
    print('  Group (3 labels):     ', results['majority']['group'])
    print('  Individual (6 labels):', results['majority']['individual'])
    print('  All (9 labels):       ', results['majority']['all'])
    print()
    print('Label occurrence stats (on raw labels):')
    print('  Group:     ', results['presence']['group'])
    print('  Individual:', results['presence']['individual'])
    print('  All:       ', results['presence']['all'])


if __name__ == '__main__':
    # Set seed=None if you want non-deterministic tie-breaking each run
    out = compute_kernel_label_statistics(root_dir='./dataset', seed=0)
    pretty_print(out)




'''
Single label distribution (majorities over raw labels):
  Group (3 labels):      {0: 15.267857142857142, 1: 67.41071428571429, 2: 17.321428571428573}
  Individual (6 labels): {0: 7.648809523809524, 1: 84.82142857142857, 2: 7.529761904761905}
  All (9 labels):        {0: 6.101190476190476, 1: 87.41071428571429, 2: 6.488095238095238}

Label occurrence stats (on raw labels):
  Group:      {'0_or_2': 68.42261904761905, 'only_1': 31.577380952380953}
  Individual: {'0_or_2': 87.02380952380952, 'only_1': 12.976190476190476}
  All:        {'0_or_2': 95.71428571428571, 'only_1': 4.285714285714286}
'''