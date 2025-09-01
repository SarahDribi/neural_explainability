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

# Load ONNX model
def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()

# Model with tracked ReLU activations
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

# Verifier 

class Verifier:
    def __init__(self, model_path, json_config, use_gpu, timeout):
        self.model_path = model_path
        self.json_config = json_config
        self.timeout = timeout
        self.use_gpu = use_gpu
        

    def is_verified_nap(self, nap,input, label, epsilon):
        result = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            
            nap=nap,
            label=label,
            use_gpu=False,
            timeout=self.timeout
            
    
        )
        return not result

"This function finds the maximum region for wich the nap holds as a spec"

def find_max_epsilon_around_input(x, nap, label, verifier, low=0.05, high=1, tol=0.001, max_iter=40):
    
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.is_verified_nap(nap,x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.is_verified_nap(nap,x, label, low):
        print("[WARN]Sorry to tell that it was not found to be robust even here {low}")
        return low*(-1)
    return low

# Greedy NAP Coarsening

def make_abstract(nap, neuron):
    layer_idx, idx = neuron
    nap_copy = copy.deepcopy(nap)
    nap_copy[layer_idx][idx] = -1
    return nap_copy


def get_neurons(nap):
    neurons = []
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            neurons.append((i, j))
    return neurons
def get_shuffeled_neurons(nap):
    neurons= get_neurons(nap)
    res=random.shuffle(neurons)
    return res

def shorten_nap_around_input(nap,input, label, epsilon, order_neurons_func, verifier,num_to_coarsen):
    new_eps=epsilon
    if not verifier.is_verified_nap(nap,input, label, new_eps):
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap)
    count=0
    for neuron in order_neurons_func(neurons):
        if count==num_to_coarsen:# first limit on coarsening process
            return nap
        a,b=neuron
        count+=1
        modified_nap = make_abstract(nap, neuron)
        if verifier.is_verified_nap(modified_nap,input, label, epsilon):
            nap = modified_nap
            print(f"[Coarsening ]Coarsening Neuron ({a}{b}) worked")

    return nap

#For now its the identity
def simple_order_neurons(neurons):
    return neurons


# Main Execution

def get_label_input(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    for x, y in dataset:
        if y==label:
            return x

def get_label_nap(mined_Naps, label):
    nap = mined_Naps[label]
    print(f"[DEBUG] Using NAP for label {label}")
    print(f"[DEBUG] NAP has {len(nap)} relu hidden layers")
    for i, layer_nap in enumerate(nap):
        print(f"[DEBUG]   NAP relu Layer {i} has = {len(layer_nap)} activations status")
    return nap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAP Coarsening Process")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds.')
    parser.add_argument('--nap_file', type=str, default="mined_naps_small_net.json", help='NAP file (JSON).')

    args = parser.parse_args()


    #pick an x from the label data
    label=args.label
    image_input=get_label_input(label)
   
    refined_nap=nap_extraction_from_onnx(args.model,image_input)
    print(display_nap_array(refined_nap))
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
    
    nap_file=args.nap_file
    print("[Info] Initial NAP:")

    with open(nap_file, "r") as f:
            naps = json.load(f)
    
    
    """
    #nap = get_label_nap(naps, label)
    #now I AM GOING TO USE THE CLASS nAP
    #refined_nap=get_label_nap(naps,label)
    
    """
    """
    max_epsilon = find_max_epsilon_around_input(image_input, refined_nap, label, verifier)
    
    
    print(f"[Info] Maximum robust epsilon for label is : {max_epsilon}")
    table=[0 for i in range(10)]
    for label in range(10):
        
        print(f"\n[INFO] Verifying label {label}")
        image_input = get_label_input(label)

        refined_nap=get_label_nap(naps,label)
        #refined_nap = nap_extraction_from_onnx(args.model, image_input)
        verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
        max_epsilon = find_max_epsilon_around_input(image_input, refined_nap, label, verifier)
        table[label]=max_epsilon
    
    """

    
    label=1
    image_input = get_label_input(label)

    refined_nap=get_label_nap(naps,label)
    refined_nap = nap_extraction_from_onnx(args.model, image_input)
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
    max_epsilon = find_max_epsilon_around_input(image_input, refined_nap, label, verifier)
    print(f"[INFO] Maximum robust epsilon for label {label} is : {max_epsilon}")
        



    
    if max_epsilon>0:
    
        num_to_coarsen=300
        print(f"I want to coarsen {num_to_coarsen}neurons\n")
        shortened_nap = shorten_nap_around_input(refined_nap,image_input, label, max_epsilon, simple_order_neurons, verifier,800)
        
        
        diff=diff_naps(refined_nap,shortened_nap)
        print(f"[Info] Coarsened neurons number was  : {diff}")
        coarsening_percentage = get_coarsening_percentage(refined_nap, shortened_nap)
        print(f"[Info] Coarsening achieved: {coarsening_percentage:.2f}% remaining neurons")

        nap_difference = diff_naps(refined_nap, shortened_nap)
        print(f"[Info] Number of neurons abstracted: {nap_difference}")
    else:
      print(f"It was not robust in the first place")
    