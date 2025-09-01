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

def get_shuffled_neurons(neurons):
    import random
    random.shuffle(neurons)
    return neurons

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
import numpy as np

def sample_nap(nap, probability):
    copy = [layer.copy() for layer in nap]  # deep copy
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if np.random.rand() < probability:
                copy[i][j] = -1  # abstract neuron with probability theta
    return copy

# I am writing another sample neurons function 
# I should write another sampling function

def sample_nap_active_neurons(nap,probability):
    copy = [layer.copy() for layer in nap]  # deep copy
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if copy[i][j]==1:
            #np.random.rand() < probability
                copy[i][j] = -1  # abstract neuron with probability theta
    return copy



def stochasticShorten(nap, input, label, epsilon, verifier, theta, max_iterations=6):
    initial_nap = [layer.copy() for layer in nap]  # deep copy
    if not verifier.is_verified_nap(nap, input, label, epsilon):
        print("[INFO] NAP is not robust initially.")
        return None  # initial NAP is not verified, no need to shorten

    for it in range(max_iterations):
        sampled_nap = sample_nap(nap, probability=theta)
        if verifier.is_verified_nap(sampled_nap, input, label, epsilon):
            print(f"[INFO] Coarsened NAP at iteration {it} was verified as robust.")
            nap = sampled_nap  # accept the coarsened NAP

    return nap

def stochasticShortenActive(nap, input, label, epsilon, verifier, theta, max_iterations=10):
    initial_nap = [layer.copy() for layer in nap]  # deep copy
    if not verifier.is_verified_nap(nap, input, label, epsilon):
        print("[INFO] NAP is not robust initially.")
        return None  # initial NAP is not verified, no need to shorten

    for it in range(max_iterations):
        sampled_nap = sample_nap_active_neurons(nap, probability=theta)
        if verifier.is_verified_nap(sampled_nap, input, label, epsilon):
            print(f"[INFO] Coarsened NAP at iteration {it} was verified as robust.")
            nap = sampled_nap  # accept the coarsened NAP

    return nap



def save_nap_to_json(nap, filename):
    with open(filename, 'w') as f:
        json.dump(nap, f)
    print(f"[INFO] Coarsened NAP saved to {filename}")
#For now its the identity
def simple_order_neurons(neurons):
    return neurons

def last_layers_first_order(neurons):
    # inverser l ordre 
    return neurons[:-1]
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
    #print(f"[DEBUG] NAP has {len(nap)} relu hidden layers")
    for i, layer_nap in enumerate(nap):
        #print(f"[DEBUG]   NAP relu Layer {i} has = {len(layer_nap)} activations status")
        continue
    return nap




def nap_active_size(nap):
    count = 0
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if nap[i][j] == 1:
                count += 1
    return count






if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="NAP Coarsening Process")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds.')
    parser.add_argument('--nap_file', type=str, default="mined_naps_small_net.json", help='NAP file (JSON).')
    parser.add_argument('--theta', type=float, default=0.2, help='Probability of abstracting each neuron during stochastic coarsening.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of stochastic coarsening iterations.')
 

    args = parser.parse_args()
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
    theta=0.1
    results = {}

for label in range(1):
    print(f"\n[INFO] Processing label {label}...")

    image_input = get_label_input(label)
    refined_nap = nap_extraction_from_onnx(args.model, image_input)
    size= nap_active_size(refined_nap)
    print(f"Size of active neurons Nap for label  {label}is {size} ")
    
    
    epsilon = -1 # Whole input space
    iterations=10


    #coarsened = stochasticShortenActive(refined_nap, image_input, label, epsilon, verifier, theta=args.theta, max_iterations=iterations)
    num_to_coarsen=300  
    coarsened=stochasticShorten(refined_nap, image_input, label, epsilon, verifier, theta, iterations)
    #shorten_nap_around_input(refined_nap,image_input, label, epsilon, get_shuffled_neurons, verifier,num_to_coarsen)
    if coarsened is None:
        print(f"[WARN] Label {label} was not robust initially. Skipping save.")
        results[label] = {
            "coarsened_neurons": None,
            "coarsening_percentage": None,
            "status": "Not robust initially"
        }
        continue
    coarsened_size=nap_active_size(coarsened)
    print(f"Size of active neurons Nap for label  {label}is {size} ")
    
    print(f"Size of active coarsened  Nap for label  {label} is {coarsened_size} ")
    diff = diff_naps(refined_nap, coarsened)
    print(f"[INFO] Coarsened neurons number: {diff}")
    coarsening_percentage = get_coarsening_percentage(refined_nap, coarsened)
    print(f"[INFO] Coarsening achieved: {coarsening_percentage:.2f}% remaining neurons")
    
    """
    
    
    
    
    # Save the coarsened NAP to a file
    nap_filename = f"coarsened_nap_label_{label}.json"
    save_nap_to_json(coarsened, nap_filename)

    # Compute coarsening statistics
    diff = diff_naps(refined_nap, coarsened)
    coarsening_percentage = get_coarsening_percentage(refined_nap, coarsened)

    print(f"[INFO] Coarsened neurons number: {diff}")
    print(f"[INFO] Coarsening achieved: {coarsening_percentage:.2f}% remaining neurons")

    # Store results
    results[label] = {
        "coarsened_neurons": diff,
        "coarsening_percentage": coarsening_percentage,
        "coarsened_nap_file": nap_filename,
        "status": "Robust"
    }

# Save all results to a summary JSON
with open('coarsening_summary.json', 'w') as f:
    json.dump(results, f, indent=4)
print("[INFO] Coarsening summary saved to coarsening_summary.json")

# Optional: Plot results if at least one label was robust
robust_labels = [label for label, info in results.items() if info["coarsened_neurons"] is not None]
if robust_labels:
    coarsened_neurons = [results[label]["coarsened_neurons"] for label in robust_labels]
    coarsening_percentages = [results[label]["coarsening_percentage"] for label in robust_labels]

    plt.figure(figsize=(12, 6))
    plt.bar(robust_labels, coarsened_neurons)
    plt.xlabel('Label')
    plt.ylabel('Number of Coarsened Neurons')
    plt.title('Coarsened Neurons per Label')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(robust_labels, coarsening_percentages)
    plt.xlabel('Label')
    plt.ylabel('Remaining Neurons (%)')
    plt.title('Remaining Neurons per Label After Coarsening')
    plt.show()
else:
    print("[INFO] No robust labels found, skipping plots.")

    """
    
    
    