import numpy as np

from clean.nap_extraction.extract_nap import (TrackedActivationsModel,nap_extraction_from_onnx)






"""
The idea behind this is to do the following :
the neurons whose activation doesns t change  much in the region  should be coarsened first
aka attributed a higher score based on frenquency of activation change 
# this is the criteria of corssing the ball with the line of the neuron more or less

""" 
"""
    neurons: list of (layer_idx, neuron_idx)
    model_tracked: model that we use to track activations
    x_nap: nap of x 
    x_input: x tensor
    epsilon: epsilon of nap_augmented exclusive robust region  
"""



def sample_around_input(x_input, epsilon, noise_factor=1.0):
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=x_input.shape)
    return np.clip(x_input + noise_factor * noise, 0.0, 1.0)  

def region_insensitivity_order(neurons, model_onnx, x_nap, x_input, epsilon, max_samples=100, noise_factor=1.0): # this is a more complex heuristic than random
    change_counts = {n: 0 for n in neurons}

    for _ in range(max_samples):
        x_prime = sample_around_input(x_input, epsilon, noise_factor=noise_factor)
        x_prime_nap = nap_extraction_from_onnx(model_onnx, x_prime)

        for (layer_idx, neuron_idx) in neurons:
            if x_nap[layer_idx][neuron_idx] != x_prime_nap[layer_idx][neuron_idx]:
                change_counts[(layer_idx, neuron_idx)] += 1

    # The more the neuron change the later we want to coarsen it
    # So we order them by increasing change frequency
    sorted_neurons = sorted(neurons, key=lambda n: change_counts[n])
    return sorted_neurons

