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
from captum.attr import NeuronIntegratedGradients
import numpy as np

# Add OVAL-BaB path
sys.path.append(os.path.join(os.getcwd(), "oval-bab", "tools", "bab_tools"))
from tools.bab_tools import vnnlib_utils





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

set_seed(42) 


# Loading the  ONNX model
def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()


# Tracking  ReLU activations
class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [2, 4, 6, 8]):
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


# Load MNIST samples grouped by label
def load_mnist_samples(limit_per_class: int = 50000):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    label_to_samples = defaultdict(list)
    for x, y in dataset:
            label_to_samples[y].append(x)
    for i in range(10):
        print(f"The lenght of these labels are {len(label_to_samples[i])}")
    return label_to_samples


# Extract NAP as list of lists of 0/1
def nap_extraction_array(model: TrackedActivationsModel, x: torch.Tensor) -> List[List[int]]:
    _ = model(x.unsqueeze(0))  # Forward pass
    nap = []
    for layer_name in sorted(model.activations.keys()):
        activations = model.activations[layer_name].flatten()
        layer_nap = [1 if val.item() > 0 else 0 for val in activations]
        nap.append(layer_nap)
    return nap


# Display NAP for a single input
def display_nap_array(nap_matrix: List[List[int]]) -> str:
    return "\n".join([f"Layer {i}: {row}" for i, row in enumerate(nap_matrix)])


# Mine NAP by majority vote (delta) across samples
def mine_nap_array(model: TrackedActivationsModel, samples: List[torch.Tensor], delta: float = 0.99) -> List[List[int]]:
    counts_per_layer = []
    for idx, x in enumerate(samples):
        nap = nap_extraction_array(model, x)
        if idx == 0:
            counts_per_layer = [[val for val in layer] for layer in nap]
        else:
            for i in range(len(nap)):
                for j in range(len(nap[i])):
                    counts_per_layer[i][j] += nap[i][j]

    total = len(samples)
    nap_template = []
    for layer_counts in counts_per_layer:
        layer_nap = [
            1 if count / total >= delta else 0 if count / total <= 1 - delta else -1
            for count in layer_counts
        ]
        nap_template.append(layer_nap)
    return nap_template


# Check if a NAP matches the template
def follows_nap_array(x: torch.Tensor, model: TrackedActivationsModel, template_nap: List[List[int]]) -> bool:
    x_nap = nap_extraction_array(model, x)
    for i in range(len(template_nap)):
        for j in range(len(template_nap[i])):
            if template_nap[i][j] == -1:
                continue  # Wildcard
            if x_nap[i][j] != template_nap[i][j]:
                return False
    return True

# here I am going to use a percentage instead 
def nap_size(nap,label):
    c=0
    total=0
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            total+=1
            if nap[i][j]==1 or nap[i][j]==0:
                c+=1
    print(f"Nap size is {c} total of neurons{total}")
    return c


def test_train_data(label,s=42,train_pct=0.9, test_pct=0.1):
    label_to_samples = load_mnist_samples()
    samples = label_to_samples[label]
    total = len(samples)
    # Split according to percentages
    num_train = int(total * train_pct)
    num_test = int(total * test_pct)
    random.shuffle(samples)

    train_samples = samples[:num_train]
    test_samples = samples[num_train:num_train + num_test]
    return train_samples,test_samples

def compute_and_verify_coverage_with_confusion_count(train_pct=0.9, test_pct=0.1, delta=0.99):
    assert 0 < train_pct + test_pct <= 1.0, "Train + test percentage must be in (0, 1]"


    onnx_path = "mnist-net_256x4.onnx"
    base_model = load_model(onnx_path)
    model = TrackedActivationsModel(base_model)

    label_to_samples = load_mnist_samples()
    labelNaps = []

    print("\n[INFO] Mining NAPs par label...")
    for label in range(10):
        samples = label_to_samples[label]
        total = len(samples)
        if total < 2:
            print(f"  Label {label} has only {total} samples — skipping.")
            labelNaps.append(None)
            continue

        # Split according to percentages
        num_train = int(total * train_pct)
        num_test = int(total * test_pct)
        random.shuffle(samples)

        train_samples = samples[:num_train]
        test_samples = samples[num_train:num_train + num_test]

        # NAP mining
        nap = mine_nap_array(model, train_samples, delta=delta)
        labelNaps.append(nap)

        print(f"  Label {label}: {num_train} train / {num_test} test")

    # Save NAPs
    nap_file = "mined_naps.json"
    with open(nap_file, "w") as f:
        json.dump(labelNaps, f)

    print(f"\n[INFO] Mined NAPs saved to {nap_file}")

    # Verify NAP coverage
    print("\n[INFO] Vérification de la couverture NAP et confusion :")
    for label in range(10):
        nap = labelNaps[label]
        if nap is None:
            print(f"Label {label}: NAP not defined.")
            continue

        samples = label_to_samples[label]
        total = len(samples)
        num_train = int(total * train_pct)
        num_test = int(total * test_pct)
        test_samples = samples[num_train:num_train + num_test]

        correct = sum(follows_nap_array(x, model, nap) for x in test_samples)

        confused = 0
        for other_label in range(10):
            if other_label == label:
                continue
            other_samples = label_to_samples[other_label]
            num_other_test = int(len(other_samples) * test_pct)
            confused += sum(
                follows_nap_array(x, model, nap)
                for x in other_samples[:num_other_test]
            )
        
        nap_size(nap,label)
        print(f"Label {label}: {correct}/{len(test_samples)} matched | {confused} confused samples from other labels")
        





if __name__ == "__main__":
    compute_and_verify_coverage_with_confusion_count(train_pct=0.7, test_pct=0.3, delta=0.95)
    
