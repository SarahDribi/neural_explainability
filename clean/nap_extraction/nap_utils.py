
from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import numpy as np

 


# Displaying a  neural activation pattern (NAP) as a matrix

def display_nap_array(nap_matrix: List[List[int]]) -> str:
    return "\n".join([f"Layer {i}: {row}" for i, row in enumerate(nap_matrix)])

# Difference and Coarsening Metrics

def diff_naps(initial_nap, second_nap):
    diff_count = 0
    for i in range(len(initial_nap)):
        for j in range(len(initial_nap[i])):
            if initial_nap[i][j] != second_nap[i][j]:
                diff_count += 1
    return diff_count


def get_coarsening_percentage(original_nap, coarsened_nap):
    total = sum(len(layer) for layer in original_nap)
    remaining = sum(1 for i in range(len(coarsened_nap)) for j in range(len(coarsened_nap[i])) if coarsened_nap[i][j] in [0, 1])
    return 100 * remaining / total

import numpy as np

import numpy as np

def summarize_nap(nap, heuristic_name="Simple order heuristic"):
    """
    Résume un Neural Activation Pattern (NAP) multi-couches.
    
    Convention :
        0 = actif
        1 = inactif
       -1 = abstrait (non fixé)
    """
    total_all = 0
    active_all = 0
    inactive_all = 0
    abstract_all = 0

    print(f"=== Neural Activation Pattern Summary ({heuristic_name}) ===")
    print(nap)

    for i, layer in enumerate(nap):
        arr = np.array(layer, dtype=int)
        total = len(arr)
        active = int(np.sum(arr == 0))
        inactive = int(np.sum(arr == 1))
        abstract = int(np.sum(arr == -1))

        total_all += total
        active_all += active
        inactive_all += inactive
        abstract_all += abstract

        print(f"\nLayer {i+1}:")
        print(f" - Total neurons   : {total}")
        print(f" - Active neurons  : {active} ({(active/total)*100:.1f}%)")
        print(f" - Inactive neurons: {inactive} ({(inactive/total)*100:.1f}%)")
        print(f" - Abstracted      : {abstract} ({(abstract/total)*100:.1f}%)")
        print(f" - Active indices  : {np.where(arr == 0)[0].tolist()}")
        print(f" - Inactive indices: {np.where(arr == 1)[0].tolist()}")
        print(f" - Abstract indices: {np.where(arr == -1)[0].tolist()}")

    if total_all > 0:
        print("\n=== Global Summary ===")
        print(f"Total neurons   : {total_all}")
        print(f"Active neurons  : {active_all} ({(active_all/total_all)*100:.1f}%)")
        print(f"Inactive neurons: {inactive_all} ({(inactive_all/total_all)*100:.1f}%)")
        print(f"Abstracted      : {abstract_all} ({(abstract_all/total_all)*100:.1f}%)")
