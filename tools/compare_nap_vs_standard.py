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

# Load ONNX model
def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()

# Model with tracked ReLU activations
class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [2, 4,6,8]):
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


def get_n_images_correctly_classified(model, label, n=5):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    correct_images = []
    for x, y in dataset:
        if y == label:
            x = x.view(-1)
            pred = model(x.unsqueeze(0)).argmax().item()
            if pred == label:
                correct_images.append(x)
                if len(correct_images) == n:
                    return correct_images
    raise ValueError(f"Could not find {n} correctly classified images for label {label}")

def run_comparison(model_path, json_config, epsilon=0.03, use_gpu=False, timeout=300, n_per_label=3):
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct

  

    results = []
    for label in range(10):
        
        images = get_n_images_correctly_classified(model, label, n_per_label)
        for idx, img in enumerate(images):
            print(f"\n=== [Label {label}] Image {idx} ===")
            try:
                std_result = verify_robustness_around_input(model_path, img, epsilon, json_config, label, use_gpu, timeout)
                nap=nap_extraction_from_onnx(model_path,img)
                nap_result = verify_nap_property_around_input(model_path, img, epsilon, json_config, nap, label, use_gpu, timeout)
            except Exception as e:
                print(f"[ERROR] During verification: {e}")
                std_result = nap_result = "ERROR"

            results.append({
                "label": label,
                "index": idx,
                "epsilon_ball_robust": std_result == False,  # UNSAT => Robust
                "nap_augmented_robust": nap_result == False,
                "is_same_result": (std_result == nap_result)
            })

    print("\n===== Résumé =====")
    for row in results:
        print(row)

    return results


        
if __name__ == '__main__':
    run_comparison(
    model_path="tools/mnist-net_256x4.onnx",
    json_config="bab_configs/mnistfc_vnncomp21.json",
    epsilon=0.03,
    use_gpu=False,
    timeout=300,
    n_per_label=10
)

# Pour ces tests choisir 20 images bien  classifiées au hasard (random shuffling)
# si c est epsilon ball robust mais pas Nap robust pour le meme epsilon
# retourne warning quelque chose de louche car les input suivant un nap sont inclus dans l epsilon ball par construction 