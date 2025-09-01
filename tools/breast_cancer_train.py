# -*- coding: utf-8 -*-
# Script complet WBC (UCI) - 2 logits + CrossEntropy + MinMax + métriques + sauvegarde
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict  # pour les helpers "MNIST-like"

# ---------------------------
# 0) Reproductibilité
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# 1) Chargement & nettoyage
# ---------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
COLS = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
        "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
        "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
FEATURE_COLS = ["clump_thickness","uniformity_cell_size","uniformity_cell_shape",
                "marginal_adhesion","single_epithelial_cell_size","bare_nuclei",
                "bland_chromatin","normal_nucleoli","mitoses"]

df = pd.read_csv(URL, names=COLS)

# Handling missing values
df["bare_nuclei"].replace("?", pd.NA, inplace=True)
df = df.dropna().copy()
df["bare_nuclei"] = df["bare_nuclei"].astype(int)

# Labels: {2:benign->0, 4:malignant->1}
df["label01"] = df["class"].map({2:0, 4:1}).astype(np.int64)

# Numpy -> Torch
X_np = df[FEATURE_COLS].astype(np.float32).values        # [N,9]
y_np = df["label01"].astype(np.int64).values             # [N]
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# ---------------------------
# 2) Split train/val (correct)
# ---------------------------
N = X.shape[0]
train_ratio = 0.8
n_train = int(train_ratio * N)

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(N, generator=g)
train_idx = perm[:n_train]
val_idx   = perm[n_train:]

train_X, train_y = X[train_idx], y[train_idx]
val_X,   val_y   = X[val_idx],   y[val_idx]

# ---------------------------
# 3) Min–Max scaler (fit sur train)
# ---------------------------
class MinMaxTransform(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.register_buffer("x_min", torch.empty(0))
        self.register_buffer("x_max", torch.empty(0))
        self.eps = eps
    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        self.x_min = X.min(dim=0).values
        self.x_max = X.max(dim=0).values
    def forward(self, x: torch.Tensor):
        denom = (self.x_max - self.x_min).clamp_min(self.eps)
        return ((x - self.x_min) / denom).clamp(0.0, 1.0)

minmax = MinMaxTransform()
minmax.fit(train_X)

# ---------------------------
# 4) Dataset & DataLoader
# ---------------------------
class TransformDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]  # y en int64 pour CrossEntropyLoss

train_ds = TransformDataset(train_X, train_y, transform=minmax)
val_ds   = TransformDataset(val_X,   val_y,   transform=minmax)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ---------------------------
# 5) Modèle à 2 logits
# ---------------------------
class WBCBinaryClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # 2 logits
        )
    def forward(self, x):
        return self.net(x)

model = WBCBinaryClassifier(input_dim=9, hidden=32)

# ---------------------------
# 6) Critère avec poids de classes (sur train uniquement)
#    Poids ~ inverse de la fréquence: w_c = N/(2*n_c)
# ---------------------------
n_pos = (train_y == 1).sum().item()
n_neg = (train_y == 0).sum().item()
n_tot = len(train_y)
w0 = n_tot / (2.0 * max(n_neg, 1))
w1 = n_tot / (2.0 * max(n_pos, 1))
class_weights = torch.tensor([w0, w1], dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------------------------
# 7) Évaluation: acc / prec / rec / F1 + confusion
# ---------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for xb, yb in loader:
        logits = model(xb)              # [B,2]
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)    # [B]
        correct += (preds == yb).sum().item()
        total_loss += loss.item() * yb.size(0)
        total += yb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def detailed_metrics(loader):
    model.eval()
    TP=FP=TN=FN=0
    for xb, yb in loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)
        y = yb
        TP += ((pred==1)&(y==1)).sum().item()
        TN += ((pred==0)&(y==0)).sum().item()
        FP += ((pred==1)&(y==0)).sum().item()
        FN += ((pred==0)&(y==1)).sum().item()
    acc  = (TP+TN)/max(TP+TN+FP+FN,1)
    prec = TP/max(TP+FP,1)
    rec  = TP/max(TP+FN,1)
    f1   = 2*prec*rec/max(prec+rec,1e-12)
    cm   = {"TN":TN, "FP":FP, "FN":FN, "TP":TP}
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "cm":cm}

# ---------------------------
# 8) Entraînement
# ---------------------------
EPOCHS = 50
for ep in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if ep % 10 == 0 or ep in (1, EPOCHS):
        vl, va = evaluate(val_loader)
        m = detailed_metrics(val_loader)
        print(f"Epoch {ep:03d} | val_loss={vl:.4f} | acc={va:.4f} | "
              f"F1={m['f1']:.4f} | P={m['prec']:.4f} | R={m['rec']:.4f} "
              f"| CM={m['cm']}")

# ---------------------------
# 9) Sauvegarde (modèle + scaler)
# ---------------------------
SAVE_PATH = "wbc_model_minmax_2logits.pt"
torch.save({
    "model_state": model.state_dict(),
    "x_min": minmax.x_min,
    "x_max": minmax.x_max,
    "feature_cols": FEATURE_COLS,
    "class_weights": class_weights
}, SAVE_PATH)
print(f"Artifacts saved to: {os.path.abspath(SAVE_PATH)}")

# ---------------------------
# 10) Exemple d'inférence sur le set de validation
# ---------------------------
@torch.no_grad()
def predict_proba(x_batch):
    model.eval()
    logits = model(x_batch)                  # [B,2]
    probs = torch.softmax(logits, dim=1)     # [B,2]
    return probs

# Prend 5 échantillons du val set
idx = torch.arange(len(val_ds))[:5]
xb = torch.stack([val_ds[i][0] for i in idx], dim=0)  # [5,9] déjà minmax
yb = torch.tensor([val_ds[i][1].item() for i in idx], dtype=torch.long)  # [5]

probs = predict_proba(xb)
preds = probs.argmax(dim=1)
print("Sample probs (columns: class0=benign, class1=malignant):")
print(probs)
print("True labels:", yb.tolist())
print("Pred labels:", preds.tolist())

# ----------------------------------------------------------------
# API analogue à MNIST, sans cache : recharge/retourne les échantillons à la demande
# ----------------------------------------------------------------
def _load_clean_wbc_df_no_cache():
    """Recharge et nettoie le WBC (aucun cache)."""
    URL_local = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    COLS_local = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
                  "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
                  "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
    df_local = pd.read_csv(URL_local, names=COLS_local)
    df_local["bare_nuclei"].replace("?", pd.NA, inplace=True)
    df_local = df_local.dropna().copy()
    df_local["bare_nuclei"] = df_local["bare_nuclei"].astype(int)
    df_local["label01"] = df_local["class"].map({2:0, 4:1}).astype(np.int64)
    return df_local

def _maybe_scale(x: torch.Tensor, minmax: 'MinMaxTransform' | None, use_scaled: bool):
    if use_scaled and (minmax is not None):
        return minmax(x)
    return x

def get_label_input_wbc(label: int, use_scaled: bool = True, minmax: MinMaxTransform | None = None) -> torch.Tensor:
    """
    Retourne 1 échantillon (tensor [9]) d'une classe donnée (0=benign, 1=malignant),
    aléatoirement, optionnellement min–max scalé avec 'minmax'.
    (Recharge le dataset; pas de cache.)
    """
    df_local = _load_clean_wbc_df_no_cache()
    sub = df_local[df_local["label01"] == int(label)]
    if sub.empty:
        raise ValueError(f"Aucun échantillon pour le label {label}.")
    row = sub.sample(n=1, random_state=random.randrange(10_000))
    x = torch.tensor(row[FEATURE_COLS].astype(np.float32).values[0], dtype=torch.float32)
    return _maybe_scale(x, minmax, use_scaled)  # shape [9]

def load_wbc_samples(limit_per_class: int = 50000, use_scaled: bool = True,
                     minmax: MinMaxTransform | None = None):
    """
    Retourne un dict: label -> list[tensor[9]], comme pour MNIST.
    (Recharge le dataset; pas de cache.)
    """
    df_local = _load_clean_wbc_df_no_cache()
    label_to_samples = defaultdict(list)
    for lbl in (0, 1):
        sub = df_local[df_local["label01"] == lbl]
        if limit_per_class is not None:
            sub = sub.head(limit_per_class)
        X_local = torch.tensor(sub[FEATURE_COLS].astype(np.float32).values, dtype=torch.float32)  # [K,9]
        if use_scaled and (minmax is not None):
            X_local = minmax(X_local)
        for i in range(X_local.shape[0]):
            label_to_samples[lbl].append(X_local[i])
    return label_to_samples

def get_label_inputs_wbc(label: int, use_scaled: bool = True,
                         minmax: MinMaxTransform | None = None):
    """
    Retourne tous les échantillons d'un label (list[tensor[9]]).
    (Recharge le dataset; pas de cache.)
    """
    df_local = _load_clean_wbc_df_no_cache()
    sub = df_local[df_local["label01"] == int(label)]
    X_local = torch.tensor(sub[FEATURE_COLS].astype(np.float32).values, dtype=torch.float32)
    if use_scaled and (minmax is not None):
        X_local = minmax(X_local)
    return [X_local[i] for i in range(X_local.shape[0])]
input = torch.randn(1, 9, dtype=torch.float32)  # batch=1, 9 features
# Export to ONNX
torch.onnx.export(
    model, 
    input, 
    "breast_cancer_model.onnx",
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},
    opset_version=11
)

print("Exported to 'breast_cancer_model.onnx'")
