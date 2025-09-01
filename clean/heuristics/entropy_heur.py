import numpy as np
# here I ll sample an element that has the same label as the input

from clean.nap_extraction.extract_nap import nap_extraction_from_onnx

from collections import defaultdict
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import random

import numpy as np


def get_neurons(nap):
    neurons = []
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            neurons.append((i, j))
    return neurons
def get_label_input(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    label_samples = [x for x, y in dataset if y == label]
    return random.choice(label_samples)

def sample_label_image(label):
    return get_label_input(label)

def entropy_heuristic(neurons, model_onnx, label, max_samples=40):
    
    counts_active = {n: 0 for n in neurons}
    total = 0

    for _ in range(max_samples):
        x_prime = sample_label_image(label)
        
        x_prime_nap = nap_extraction_from_onnx(model_onnx, x_prime) 

        for (layer_idx, neuron_idx) in neurons:
            activation = x_prime_nap[layer_idx][neuron_idx]
            status = 1 if activation == 1  else 0
            counts_active[(layer_idx, neuron_idx)] += status
        total += 1

    
    eps = 1e-12
    def H(p):
        p = min(max(p, eps), 1.0 - eps)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    entropies = {n: H(counts_active[n] / max(1, total)) for n in neurons}
    # plus stable = entropie faible, donc on veut coarsen en premier les neurones avec entropie faible
    sorted_neurons = sorted(neurons, key=lambda n: entropies[n])
    return sorted_neurons,entropies





if __name__ == "__main__":
    input=get_label_input(5)  # Example input, replace with actual input
    onnx_path = "mnist-10x2.onnx"  # Replace with
  # Example neurons, replace with actual NAP extraction
    model_onnx = "mnist-10x2.onnx"  # Replace with your ONNX model
    nap= nap_extraction_from_onnx(model_onnx, input)  # Extract NAP from the model and input
    neurons =get_neurons(nap)
   
    label = 5  # Example label
    ordering,entropies = entropy_heuristic(neurons, model_onnx, label)
    print("Neuron ordering based on entropy heuristic:", ordering,entropies)
# This function orders neurons based on their entropy, which is a measure of uncertainty in their activation
# The lower the entropy, the more certain the neuron is about its activation, hence it is coarsened first.


# I need to cache the neurons entropy to avoid recomputing it
# This can be done by storing the entropies in a dictionary 