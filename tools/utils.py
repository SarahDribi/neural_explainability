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


# Add OVAL-BaB path
sys.path.append(os.path.join(os.getcwd(), "oval-bab", "tools", "bab_tools"))
from tools.bab_tools import vnnlib_utils


# Load ONNX model
def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()

# Model with tracked ReLU activations
class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [2,4]):
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

def get_label_inputs(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    l=[]
    for x, y in dataset:
        if y==label:
            l.append(x)
    return l
#extract nap from tracked model

def nap_extraction(model: TrackedActivationsModel, x: torch.Tensor) -> List[List[int]]:
    _ = model(x.unsqueeze(0))
    nap = []
    for layer_name in sorted(model.activations.keys()):
        activations = model.activations[layer_name].flatten()
        layer_nap = [1 if val.item() > 0 else 0 for val in activations]
        nap.append(layer_nap)
    return nap

# Extract NAP from onnx model

def nap_extraction_from_onnx(onnx_model_path,x):
    base_model = load_model(onnx_model_path)
    model_tracked = TrackedActivationsModel(base_model)
    nap=nap_extraction(model_tracked,x)
    return nap

# my intuituion here is to extract the nap  from the nearest 
# neighbors of that class
def k_nearest_neighbors_nap_extract():
    return




# Display NAP

def display_nap_array(nap_matrix: List[List[int]]) -> str:
    return "\n".join([f"Layer {i}: {row}" for i, row in enumerate(nap_matrix)])

# Difference and Coarsening Metrics

def diff_naps(first_nap, second_nap):
    diff_count = 0
    for i in range(len(first_nap)):
        for j in range(len(first_nap[i])):
            if first_nap[i][j] != second_nap[i][j]:
                diff_count += 1
    return diff_count


def get_coarsening_percentage(original_nap, coarsened_nap):
    total = sum(len(layer) for layer in original_nap)
    remaining = sum(1 for i in range(len(coarsened_nap)) for j in range(len(coarsened_nap[i])) if coarsened_nap[i][j] in [0, 1])
    return 100 * remaining / total

if __name__ == '__main__':
    print("h")

