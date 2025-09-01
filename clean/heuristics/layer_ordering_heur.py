import numpy as np
# here I ll sample an element that has the same label as the input

from clean.nap_extraction.extract_nap import nap_extraction_from_onnx

from collections import defaultdict
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import random

import numpy as np
from clean.coarsening.nap_coarsen import get_neurons

def ascending_layer_ordering_heur(neurons):     
    """
    This function takes a list of neurons and returns them ordered by their layer index in ascending order.
    Each neuron is represented as a tuple (layer_index, neuron_index).
    """
    return sorted(neurons, key=lambda x: x[0])



def descending_layer_ordering_heur(neurons):
    """
    This function takes a list of neurons and returns them ordered by their layer index in descending order.
    Each neuron is represented as a tuple (layer_index, neuron_index).
    """
    return sorted(neurons, key=lambda x: x[0], reverse=True)

def get_label_input(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    label_samples = [x for x, y in dataset if y == label]
    return random.choice(label_samples)




if __name__ == "__main__":
    input=get_label_input(5) 
    onnx_path = "mnist-10x2.onnx"  
    model_onnx = "mnist-10x2.onnx"  
    nap= nap_extraction_from_onnx(model_onnx, input)  
    neurons =get_neurons(nap)
   
    label = 5  # Example label
    ordering_first= ascending_layer_ordering_heur(neurons)
    ordering_second = descending_layer_ordering_heur(neurons)
    print("Neuron ordering in ascending layer order:", ordering_first)
    print("Neuron ordering in descending layer order:", ordering_second)
    