import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
from pathlib import Path
import os
import random


# Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (optionnel) pour des convolutions/BLAS déterministes :
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
_COLS = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
         "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
         "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
_FEATURE_COLS = ["clump_thickness","uniformity_cell_size","uniformity_cell_shape",
                 "marginal_adhesion","single_epithelial_cell_size","bare_nuclei",
                 "bland_chromatin","normal_nucleoli","mitoses"]

def load_data_set():
    df = pd.read_csv(_URL, names=_COLS)
    df = df.replace("?", np.nan).dropna()
    df[_FEATURE_COLS] = df[_FEATURE_COLS].astype(float)
    df["label"] = df["class"].map({2: 0, 4: 1}).astype(np.int64)

    X = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y

def split_then_normalize(X, y, test_size=0.2, seed=42, return_scaler=True):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True
    )
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    Xtr_t = torch.from_numpy(X_tr_sc)
    ytr_t = torch.from_numpy(y_tr).long()
    Xte_t = torch.from_numpy(X_te_sc)
    yte_t = torch.from_numpy(y_te).long()

    if return_scaler:
        return Xtr_t, ytr_t, Xte_t, yte_t, scaler
    else:
        return Xtr_t, ytr_t, Xte_t, yte_t


class WBCBinaryClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)


# Training

def train_model(model, X_train, y_train, X_val, y_val,
                num_epochs=50, lr=1e-3, weight_decay=1e-4,
                batch_size=32, device="cpu", early_stop_patience=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        avg_loss = total_loss / len(train_ds)
        acc = correct / len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device))
            val_loss = F.cross_entropy(logits, y_val.to(device)).item()
            val_acc = (logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:03d}] Train loss {avg_loss:.4f}, acc {acc:.3f} | "
                  f"Val loss {val_loss:.4f}, acc {val_acc:.3f}")

        # Early stopping (simple)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# Eval helpers

@torch.no_grad()
def predict_proba(model, X, device="cpu"):
    model.eval()
    logits = model(X.to(device))
    probs = F.softmax(logits, dim=1)[:, 1]  # probabilité de la classe 1 (malignant)
    return probs.cpu().numpy()

@torch.no_grad()
def predict(model, X, device="cpu", threshold=0.5):
    p = predict_proba(model, X, device=device)
    return (p >= threshold).astype(np.int64)


# Main

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Chargement et split
    X, y = load_data_set()
    X_train, y_train, X_test, y_test, scaler = split_then_normalize(X, y, test_size=0.2, seed=42, return_scaler=True)

    # Sauvegarde du scaler
    joblib.dump(scaler, "artifacts/breast_minmax_scaler.joblib")

    # Modèle
    model = WBCBinaryClassifier(input_dim=9, hidden=32)

    # Entraînement
    model = train_model(
        model, X_train, y_train, X_test, y_test,
        num_epochs=200, lr=1e-3, weight_decay=1e-4,
        batch_size=32, device=device, early_stop_patience=20
    )

    # Évaluation (seuil 0.5 par défaut)
    y_proba = predict_proba(model, X_test, device=device)
    y_pred = (y_proba >= 0.5).astype(int)

    print("\nClassification report (test):")
    print(classification_report(y_test.numpy(), y_pred, digits=4))

    cm = confusion_matrix(y_test.numpy(), y_pred)
    print("Confusion matrix:\n", cm)

    # AUC (attention: nécessite au moins une pos/neg dans le split)
    try:
        auc = roc_auc_score(y_test.numpy(), y_proba)
        print(f"AUC: {auc:.4f}")
    except ValueError as e:
        print("AUC non calculable:", e)

    # Sauvegarde du modèle
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), os.path.join("artifacts", "wbc_mlp.pt"))
    print("\nModèle sauvegardé dans artifacts/wbc_mlp.pt et scaler dans breast_minmax_scaler.joblib")

import torch.onnx

model = WBCBinaryClassifier()
model.load_state_dict(torch.load("artifacts/wbc_mlp.pt", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 9)  # (batch, 9)

torch.onnx.export(
    model,
    dummy_input,
    "artifacts/wbc_mlp.onnx",
    input_names=["input"],
    output_names=["output"],  
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11,
)
print("Exported to 'artifacts/wbc_mlp.onnx'")






# --- Helpers WBC (analogue to  MNIST) ---
from collections import defaultdict
from typing import Dict, List, Optional
import joblib
import os

def _load_and_scale_WBC(scaler_path: str = "artifacts/breast_minmax_scaler.joblib"):
    """Charge le dataset brut, puis applique le même MinMaxScaler que celui du training .
       Retourne X_scaled (np.float32, shape [N,9]) et y (np.int64, shape [N]).
    """
    X, y = load_data_set()
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        # Sinon dépanage  (on fit sur tout), mais garde le code robuste si le scaler n’est pas encore dumpé
        scaler = MinMaxScaler().fit(X)
    X_sc = scaler.transform(X).astype(np.float32)
    return X_sc, y

def get_label_input(label: int, scaler_path: str = "artifacts/breast_minmax_scaler.joblib") -> torch.Tensor:
    """Retourne UN échantillon 9-D (tensor float32) normalisé pour le label donné (0 ou 1)."""
    X_sc, y = _load_and_scale_WBC(scaler_path)
    idx = np.where(y == label)[0]
    if len(idx) == 0:
        raise ValueError(f"Aucun échantillon pour le label {label}.")
    i = random.choice(idx.tolist())
    return torch.from_numpy(X_sc[i])  # shape: (9,)

def load_wbc_samples(limit_per_class: Optional[int] = None,
                     scaler_path: str = "artifacts/breast_minmax_scaler.joblib") -> Dict[int, List[torch.Tensor]]:
    """Retourne {0: [tensors 9-D], 1: [tensors 9-D]} (normalisés).
       limit_per_class : si non None, échantillonne au plus ce nombre par classe.
    """
    X_sc, y = _load_and_scale_WBC(scaler_path)
    buckets: Dict[int, List[torch.Tensor]] = {0: [], 1: []}
    for xi, yi in zip(X_sc, y):
        buckets[int(yi)].append(torch.from_numpy(xi))
    if limit_per_class is not None:
        for k in buckets:
            if len(buckets[k]) > limit_per_class:
                buckets[k] = random.sample(buckets[k], limit_per_class)
    return buckets

def get_label_inputs(label: int, limit: Optional[int] = None,
                     scaler_path: str = "artifacts/breast_minmax_scaler.joblib") -> torch.Tensor:
    """Retourne un tenseur (N, 9) de tous les échantillons (ou des 'limit' premiers) pour un label."""
    X_sc, y = _load_and_scale_WBC(scaler_path)
    sel = X_sc[y == label]
    if sel.size == 0:
        raise ValueError(f"Aucun échantillon pour le label {label}.")
    if limit is not None:
        sel = sel[:limit]
    return torch.from_numpy(sel)  # shape: (N, 9)
