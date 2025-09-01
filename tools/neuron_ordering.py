# this file will be responsible for the neuron ordering strategies

from utils_verif import get_adversarial_with_bab 
# I should write this function
#use branch and bound to have an adversarial example

# I should try different orderings 

# ordering by layer Assuming that higher layers are responsible for more significant patterns 




# guiding the exploration by a heuristic based approach (statistical)
# guiding the exploration statistically



import numpy as np

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


#pruning techniques 
# distillation ?
#quantization



# Extract NAP from onnx model

def nap_extraction_from_onnx(onnx_model_path,x):
    base_model = load_model(onnx_model_path)
    model_tracked = TrackedActivationsModel(base_model)
    nap=nap_extraction(model_tracked,x)
    return nap

# using adversarial search what is called their optimistic approch 
# 
# for this part (we are just going to consider the neurons that have different behaviors in the adversarial examples as important) 


# the adversarial examples are generated with branch and bound
# wich makes it faster 
def get_different_neurons(input_nap,adv_nap):
    # assert they are of the same size
    neurons_diff=[]
    for i in range(len(input_nap)):
        for j in range(input_nap[i]):
            if input_nap[i][j]!=adv_nap[i][j]:
                # consider this neuron as important
                # add its layer index and index at layer
                neurons_diff.append((i,j))
    return neurons_diff


#n is the number of adversarial examples to generate 
def get_important_neurons_around_input(input,nap,n):
    important_neurons_around_input={}
    for i in range(n):

        adversarial_example=get_adversarial_example_around_input(input,epsilon)
        adversarial_nap=nap_extraction_from_onnx(onnx_model_path,adversarial_example)
        # see which activations were distinct from the previous ones
        # We consider that every  neuron that has changed activation status as important
        neurons_diff=get_different_neurons(nap,adversarial_nap)
        neurons_union(important_neurons_around_input,neurons_diff)
    return important_neurons_around_input


def entropy_based_neuron():
    
    
    

#
# how can i use that in my coarsening strategy ?

    
    



    




# Ideas are : 

# Slice a  subnetwork to make the coarsening faster
#To make the process faster , we can use a paper called an INN based abstraction method for large scale neural network verification 


# other ideas are welcome and a lot 
"""
modularizethe script into separate files:

nap_extraction.py

coarsening.py

verification.py

visualization.py

main.py
"""
