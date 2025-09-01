import numpy as np

from utils import TrackedActivationsModel
from utils import nap_extraction_from_onnx
def get_shuffled_neurons(neurons):
    import random
    random.shuffle(neurons)
    return neurons

# Hybrid Scoring Function
def simple_order_neurons(neurons):
    return neurons






"""
The idea behind this is to do the following :
the neurons whose activation doesns t change  much in the region  should be coarsened first
aka attributed a higher score based on frenquency of activation change 
# this is the criteria of corssing the ball with the line of the neuron more or less
a little example:

for example: I have 4 neurons 



"""
"""
def get_neurons(nap):
    neurons = []
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            neurons.append((i, j))
    return neurons
 this was how the neurons are
"""


from utils import nap_extraction
import numpy as np
import random

def sample_around_input(x_input, epsilon, noise_factor=1.0):
    # Ajoute un bruit uniforme dans la boule ε
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=x_input.shape)
    return np.clip(x_input + noise_factor * noise, 0.0, 1.0)  
"""
    neurons: list of (layer_idx, neuron_idx)
    model_tracked: model that we use to track activations
    x_nap: nap of x 
    x_input: x tensor
    epsilon: epsilon of nap_augmented exclusive robust region  
"""
def region_insensitivity_order(neurons, model_onnx, x_nap, x_input, epsilon, max_samples=100, noise_factor=1.0): # this is a more complex heuristic than random

    # A change counter
    
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

"""
This heuristic is a heuristic that makes me order the neurons given the importance 
of their preactivation

"""
def get_preactivation_derivative_for_logit(neuron ,model,label):
    return 1
def preactivation_impact_order(neurons, model_onnx, x_nap, x_input, epsilon,x_label): 

    
    
    importance_score = {n: 0 for n in neurons}
    
    
    for neuron in neurons:
            
            get_preactivation_derivative_for_logit(neuron,model_onnx,x_label)

    # The more the neuron change the later we want to coarsen it
    # So we order them by increasing change frequency
    sorted_neurons = sorted(neurons, key=lambda n: importance_score[n])
    return sorted_neurons




# Entropy based methods 


# adversarial attacks based heuristics





























# Adversarial Neuron Importance
"""




def get_different_neurons(input_nap, adv_nap):
    neurons_diff = []
    for i in range(len(input_nap)):
        for j in range(len(input_nap[i])):
            if input_nap[i][j] != adv_nap[i][j]:
                neurons_diff.append((i, j))
    return neurons_diff

def neurons_union(existing_neurons, new_neurons):
    for neuron in new_neurons:
        existing_neurons[neuron] = True
    return existing_neurons
"""
"""
def get_adversarial_example_around_input(input, epsilon, model_path, json_config, timeout=300, use_gpu=False):
    adv_example = get_adversarial_with_bab(
        model_path=model_path,
        input=input,
        epsilon=epsilon,
        json_config=json_config,
        timeout=timeout,
        use_gpu=use_gpu
    )
    return adv_example



"""

"""

"""

"""
def get_important_neurons_around_input(input, nap, n, epsilon, model_path, json_config):
    important_neurons = {}
    for i in range(n):
        adversarial_example = get_adversarial_example_around_input(input, epsilon, model_path, json_config)
        adversarial_nap = nap_extraction_from_onnx(model_path, adversarial_example)
        neurons_diff = get_different_neurons(nap, adversarial_nap)
        neurons_union(important_neurons, neurons_diff)
    return list(important_neurons.keys())





"""


"""


# Entropy-Based Neuron Importance


def compute_neuron_entropy(nap_samples):
    num_layers = len(nap_samples[0])
    neuron_entropies = []

    for layer_idx in range(num_layers):
        layer_neurons = len(nap_samples[0][layer_idx])
        layer_entropy = []

        for neuron_idx in range(layer_neurons):
            activations = [nap[layer_idx][neuron_idx] for nap in nap_samples]
            p = np.mean(activations)
            if p in [0, 1]:
                entropy = 0
            else:
                entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
            layer_entropy.append(entropy)

        neuron_entropies.append(layer_entropy)

    return neuron_entropies



def score_neuron(neuron, adversarial_importance, entropy_map, adv_weight=1.0, entropy_weight=1.0):
    adv_score = adv_weight if neuron in adversarial_importance else 0.0
    entropy = entropy_map[neuron[0]][neuron[1]]
    return adv_score - entropy_weight * entropy

def order_neurons_hybrid(neurons, adversarial_importance, entropy_map):
    scored_neurons = sorted(
        neurons,
        key=lambda n: score_neuron(n, adversarial_importance, entropy_map),
        reverse=False  # Low score = coarsen first
    )
    return scored_neurons


# Layer-Wise Coarsening Schedule


def layer_wise_order(neurons):
    
    layer_priority = [3,2,1,0] #prioritizes coarsening in this order
    
    return sorted(neurons, key=lambda n: (layer_priority.index(n[0]), n[1]))


# Random Shuffling
#
def order_based_on_entropy(neurons):
    def compute_neuron_entropy(nap_samples):
        num_layers = len(nap_samples[0])
        neuron_entropies = []

        for layer_idx in range(num_layers):
            layer_neurons = len(nap_samples[0][layer_idx])
            layer_entropy = []

            for neuron_idx in range(layer_neurons):
                activations = [nap[layer_idx][neuron_idx] for nap in nap_samples]
                p = np.mean(activations)
                if p in [0, 1]:
                    entropy = 0
                else:
                    entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
                layer_entropy.append(entropy)

            neuron_entropies.append(layer_entropy)

        return neuron_entropies
    # sort these neurons based on entropy , the lower the entropy the more decisive it is
    

"""

























