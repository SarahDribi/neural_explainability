from collections import defaultdict
import random
import numpy as np
import pandas as pd
import torch

# 
_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
_COLS = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
         "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
         "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
_FEATURE_COLS = ["clump_thickness","uniformity_cell_size","uniformity_cell_shape",
                 "marginal_adhesion","single_epithelial_cell_size","bare_nuclei",
                 "bland_chromatin","normal_nucleoli","mitoses"]

def _load_clean_wbc_df():
    """Charge et nettoie WBC: enlève '?', map {2->0, 4->1} vers label01."""
    df = pd.read_csv(_URL, names=_COLS)
    df["bare_nuclei"].replace("?", pd.NA, inplace=True)
    df = df.dropna().copy()
    df["bare_nuclei"] = df["bare_nuclei"].astype(int)
    df["label01"] = df["class"].map({2: 0, 4: 1}).astype(np.int64)
    return df

def get_label_input(label: int):
    """
    Retourne 1 échantillon tensor[9] pour un label donné (0=benign, 1=malignant),
    choisi aléatoirement.
    """
    df = _load_clean_wbc_df()
    sub = df[df["label01"] == int(label)]
    if sub.empty:
        raise ValueError(f"Aucun échantillon pour le label {label}.")
    row = sub.sample(n=1, random_state=random.randrange(10_000))
    x = torch.tensor(row[_FEATURE_COLS].astype(np.float32).values[0], dtype=torch.float32)
    return x  # shape [9]

def load_samples(limit_per_class: int = 5000):
    """
    renvoie un dict label->list[tensor[9]].
    
    """
    df = _load_clean_wbc_df()
    label_to_samples = defaultdict(list)
    for lbl in (0, 1):
        sub = df[df["label01"] == lbl]
        if limit_per_class is not None:
            sub = sub.head(limit_per_class)
        X_local = torch.tensor(sub[_FEATURE_COLS].astype(np.float32).values, dtype=torch.float32)
        for i in range(X_local.shape[0]):
            label_to_samples[lbl].append(X_local[i])
    return label_to_samples

def get_label_inputs(label: int):
    """
    Retourne tous les échantillons d’un label (list[tensor[9]]).
    """
    df = _load_clean_wbc_df()
    sub = df[df["label01"] == int(label)]
    X_local = torch.tensor(sub[_FEATURE_COLS].astype(np.float32).values, dtype=torch.float32)
    return [X_local[i] for i in range(X_local.shape[0])]
