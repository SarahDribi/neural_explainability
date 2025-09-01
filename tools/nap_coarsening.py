# Greedy NAP Coarsening
from tqdm import tqdm # I added a progress bar to not feel lost 
import copy
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

"""
It should also take the model because the advanced versions take the model
it s used in heuristics

"""


def shorten_nap_around_input(nap, input, label, epsilon, order_neurons_func, verifier):
    nap_copy = copy.deepcopy(nap)
    new_eps = epsilon

    if not verifier.is_verified_nap(nap_copy, input, label, new_eps):
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap_copy)

    # Progress bar for the coarsening loop
    for neuron in tqdm(order_neurons_func(neurons), desc="Coarsening NAP", unit="neuron"):
        a, b = neuron
        modified_nap = make_abstract(nap_copy, neuron)

        if verifier.is_verified_nap(modified_nap, input, label, epsilon):
            nap_copy = modified_nap
            print(f"[Coarsening] Neuron ({a},{b}) coarsened successfully.")

    return nap_copy
def shorten_nap_around_input_complex_heuristic(nap,args,input, label, epsilon, order_neurons_func, verifier): # like region_insensivity_order(neurons, model_tracked, x_nap, x_input, epsilon, max_samples=100, noise_factor=1.0):
    nap_copy=copy.deepcopy(nap)
    new_eps=epsilon
    if not verifier.is_verified_nap(nap_copy,input, label, new_eps):
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap_copy)
    count=0
    for neuron in order_neurons_func(neurons,args.model,nap,input,epsilon):
        
        a,b=neuron
        modified_nap = make_abstract(nap_copy, neuron)
        if verifier.is_verified_nap(modified_nap,input, label, epsilon):
            nap_copy = modified_nap
            print(f"[Coarsening ]Coarsening Neuron ({a}{b}) worked")

    return nap_copy
import numpy as np

def sample_nap(nap, probability):
    copy = [layer.copy() for layer in nap]  # deep copy
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if np.random.rand() < probability:
                copy[i][j] = -1  # abstract neuron with probability theta
    return copy




def stochasticShorten(nap, input, label, epsilon, verifier, theta, max_iterations=35):
    initial_nap = [layer.copy() for layer in nap]  
    initial_success=0
    if not verifier.is_verified_nap(nap, input, label, epsilon):
        print("[INFO] NAP is not robust initially.")
        return None  # initial NAP is not verified, no need to shorten

    for it in range(max_iterations):
        sampled_nap = sample_nap(nap, probability=theta)
        if verifier.is_verified_nap(sampled_nap, input, label, epsilon):
            print(f"[INFO] Coarsened NAP at iteration {it} was verified as robust.")
            nap = sampled_nap 
            initial_success+=1

    return nap,initial_success

