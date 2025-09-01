import torch
import json
from tools.bab_tools import vnnlib_utils
from tools.bab_tools.bab_runner import bab_output_from_return_dict
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from utils_verif import (
    get_label_nap,
    verify_robustness_around_input,
    verify_nap_property_around_input,
    
)
import os
import sys
from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import matplotlib.pyplot as plt
import random
import json
import copy
import numpy as np
import argparse

# Add OVAL-BaB path
sys.path.append(os.path.join(os.getcwd(), "oval-bab", "tools", "bab_tools"))
from tools.bab_tools import vnnlib_utils
from utils_verif import  verify_nap_property_around_input
from utils_verif import verify_robustness_around_input

# Model with tracked ReLU activations
class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [2, 4]):
        super().__init__()
        self.base = base
        self.relu_layers = relu_layers
        self.activations = {}

    def forward(self, x):
        self.activations = {}
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.relu_layers:
                self.activations[f"relu_{i}"] = x.clone()
        return x
# Load ONNX model
def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()
# Extract NAP from onnx model
def nap_extraction(model: TrackedActivationsModel, x: torch.Tensor) -> List[List[int]]:
    _ = model(x.unsqueeze(0))
    nap = []
    for layer_name in sorted(model.activations.keys()):
        activations = model.activations[layer_name].flatten()
        layer_nap = [1 if val.item() > 0 else 0 for val in activations]
        nap.append(layer_nap)
    return nap

def nap_extraction_from_onnx(onnx_model_path,x):
    base_model = load_model(onnx_model_path)
    model_tracked = TrackedActivationsModel(base_model)
    nap=nap_extraction(model_tracked,x)
    return nap

# Model with tracked ReLU activations
class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [2, 4]):
        super().__init__()
        self.base = base
        self.relu_layers = relu_layers
        self.activations = {}

    def forward(self, x):
        self.activations = {}
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.relu_layers:
                self.activations[f"relu_{i}"] = x.clone()
        return x

# Load MNIST samples
def get_label_input(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    label_samples = [x for x, y in dataset if y == label]
    return random.choice(label_samples)


def load_mnist_samples(limit_per_class: int = 50000):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    label_to_samples = defaultdict(list)
    for x, y in dataset:
        label_to_samples[y].append(x)
    return label_to_samples


def get_correct_images_per_class(model, n_per_class=20, seed=0):
    rng = random.Random(seed)
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)

    label_to_samples = defaultdict(list)
    for x, y in dataset:
        x = x.view(-1)
        if model(x.unsqueeze(0)).argmax().item() == y:
            label_to_samples[y].append((x, y))

    for lbl in label_to_samples:
        rng.shuffle(label_to_samples[lbl])
        label_to_samples[lbl] = label_to_samples[lbl][:n_per_class]

    return [(x, y) for lbl in sorted(label_to_samples) for x, y in label_to_samples[lbl]]


def compare_20_per_class(model_path, json_cfg, eps=0.03, timeout=300, seed=0):
    model=load_model(model_path)
    samples = get_correct_images_per_class(model, n_per_class=20, seed=seed)

    print(f"\n[INFO] 20 images bien classifiées *par classe* (ε = {eps})\n")
    warnings = 0
    total = 0
    for idx, (img, lbl) in enumerate(samples, 1):
        try:
            std = verify_robustness_around_input(
                model_path, img, eps, json_cfg, lbl, False, timeout)
            nap = verify_nap_property_around_input(
                model_path, img, eps, json_cfg,
                nap_extraction_from_onnx(model_path, img),
                lbl, False, timeout)
        except Exception as e:
            print(f"[{idx:03d}] label={lbl} | ERREUR : {e}")
            continue

        std_ok  = (std is False)
        nap_ok  = (nap is False)

        line = f"[{idx:03d}] label={lbl} | std={'good' if std_ok else 'bad'} | nap={'good' if nap_ok else 'bad'}"
        if std_ok and not nap_ok:
            line += "  ←  suspect (NAP )"
            warnings += 1
        print(line)
        total += 1

    print(f"\n[SUMMARY] {warnings} / {total} cas suspects (std robuste mais NAP non‑robuste)\n")




if __name__ == "__main__" :


    compare_20_per_class(
        model_path="tools/mnist-10x2.onnx",
        json_cfg="bab_configs/mnistfc_vnncomp21.json",
        eps=0.0,
        timeout=300,
        seed=42
    )




# J'ai run des unit tests et j 'ai 
"""
1. Vérification standard (ε-ball)
Je vérifie que pour tout x′ ∈ B(x, ε), f(x′) = f(x).

Si le vérificateur dit que c’est robuste, on peut (presque toujours) faire confiance au résultat (si le vérif est complet).

Si le vérificateur dit non-robuste, il peut avoir trouvé un contre-exemple réel ou pas → incomplétude.

2. Vérification epsilon NAP-augmented
Je restreins ton espace d’entrée à x′ ∈ B(x, ε) ∩ NAP(x).

Cet espace est plus petit que la ε-ball.

(puisque NAP_augented_epsilon_region(x) ⊆ B(x, ε)).


"""
