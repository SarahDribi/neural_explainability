import copy
import random
#from adversarial_attack import adversarial_example

from nap_robustness_from_onnx import get_label_nap
from utils_verif import verify_nap_property
import json

import argparse
from cleaned import display_nap_array

# I added a verifier class to  make ii easily replaceable
#the verifier class encapsulates the model , the label , the verifiying function
"""
The verifier has an imbedded function that verifies the nap that 
I ve built seperatly in nap_robustness_from_onnx
# it should have the model 
# the nap file 
# the verification branch and bound config
# it should take as an argument a nap mined from the model   , a label , a region 
# and it should output wheter or not this nap specification for this label is 
robust within this region 
for epsilon=-1 we mean the whole possible  input space   


"""


"""
I still need to write the function that finds an adversarial example


"""


class Verifier:
    def __init__(self, model_path, json_config, use_gpu,timeout):
        self.model_path = model_path
        self.json_config = json_config
        self.timeout=timeout
        self.use_gpu = use_gpu
        

    def is_verified_nap(self, nap, label,epsilon):
       
       #Here I should replace this with a version that takes this 
       # nap instead of extractiong it from the file 
        result = verify_nap_property(
            model_path=self.model_path,
            json_config=self.json_config,
            nap=nap, 
            label=label,
            use_gpu=False,
            eps=epsilon,
            timeout=self.timeout
        )

        return not result  
    def get_adversarial_example(self,sample,epsilon):
        return sample
    



def make_abstract(nap, neuron):
    layer_idx, idx = neuron
    nap_copy = copy.deepcopy(nap)
    nap_copy[layer_idx][idx] = -1  #freeing the neuron
    return nap_copy


def get_neurons(nap):
    neurons = []
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            neurons.append((i, j))
    return neurons


# Shorten NAP  => greedy one by one neuron  coarsening 

def shorten_Nap(nap, label,epsilon, order_neurons_func, verifier):
    if verifier.is_verified_nap(nap, label,epsilon):
        print("This NAP isn't robust in the first place.")
        return None

    neurons = get_neurons(nap)

    for neuron in order_neurons_func(neurons):
        modified_nap = make_abstract(nap, neuron)


        if not verifier.is_verified_nap(modified_nap, label,epsilon):
            continue  # Skipping the abstraction if verification fails
        else:
            nap = modified_nap  # Accept the abstraction 

    return nap


# Stochast

def is_included_in_nap(neuron, nap):
    layer_idx, idx = neuron
    return nap[layer_idx][idx] in [0, 1]


def sample_NAP(nap, theta):
    nap_copy = copy.deepcopy(nap)
    for i in range(len(nap_copy)):
        for j in range(len(nap_copy[i])):
            if nap_copy[i][j] in [0, 1] and random.random() < theta:
                nap_copy[i][j] = -1
    return nap_copy


def stochasticShorten(theta,epsilon, nap, desired_size, verifier, label):
    if not verifier.is_verified_nap(nap, label,epsilon):
        return None

    neurons = get_neurons(nap)
    #On calcule le nombre des neurones qui sont raffinés 
    included_neurons = sum(1 for neuron in neurons if is_included_in_nap(neuron, nap))

    while included_neurons > desired_size:
        sampled_nap = sample_NAP(nap, theta)
        if verifier.is_verified_nap(sampled_nap, label):
            nap = sampled_nap
            included_neurons = sum(1 for neuron in get_neurons(nap) if is_included_in_nap(neuron, nap))
        else:
            break

    return nap


# Optimistic Pruning using adversarial examples
"""
L'idée de cet algo est d 'estimer que les neurones dont l activation change pour l adversarial
examples autour de l imput doivent etre considérés comme importants 
"""
def optimistic_approach(model, nap,epsilon, label_dataset, get_input_nap,verifier):
    mandatory = []
    neurons = get_neurons(nap)
    # getting an adversarial example within that region 
    for sample in label_dataset:
        adversarial_ex = verifier.adversarial_example(sample,epsilon)
        adversarial_nap = get_input_nap(model, adversarial_ex)

        for neuron in neurons:
            layer_idx, idx = neuron
            if is_included_in_nap(neuron, nap):
                if adversarial_nap[layer_idx][idx] != nap[layer_idx][idx]:
                    mandatory.append(neuron)

    return mandatory


# First intuitive neuron ordering =>  Identity

def simple_order_neurons(neurons):
    return neurons


# StochCoarsen Algorithm

def StochCoarsen(nap, mandatory_neurons, theta, desired_size, verifier, label):
    current_neurons = mandatory_neurons

    if not verifier.is_verified_nap(nap, label):
        print("The Nap wasn' t robust in the first place")
        return None

    while count_non_abstract_neurons(nap) > desired_size:
        sampled_nap = sample_NAP_from_neurons(nap, current_neurons, theta)

        if verifier.is_verified_nap(sampled_nap, label):
            found_neurons = []
            for neuron in get_neurons(nap):
                layer_idx, idx = neuron
                if sampled_nap[layer_idx][idx] == nap[layer_idx][idx]:
                    found_neurons.append(neuron)
            current_neurons = found_neurons
            nap = sampled_nap
        else:
            sampled_nap = sample_NAP_from_neurons(nap, current_neurons, theta)

    return nap





def count_non_abstract_neurons(nap):
    return sum(1 for layer in nap for neuron in layer if neuron in [0, 1])


def sample_NAP_from_neurons(nap, neurons, theta):
    nap_copy = copy.deepcopy(nap)
    for neuron in neurons:
        layer_idx, idx = neuron
        if nap_copy[layer_idx][idx] in [0, 1] and random.random() < theta:
            nap_copy[layer_idx][idx] = -1
    return nap_copy

def diff_Naps(first_nap,second_nap):
    #this function should display the difference 
    # between the first refined Nap and its coarsened version 
    return 
def get_coarsening_percentenge(first_nap, second_nap):
    #should write this 
    return 50


"""
this command line runs the coarsening process for the label one 
I should add a coarsening debug

"""
if __name__ == '__main__':
    # we run  a coarsening example on a nap 
    parser = argparse.ArgumentParser(description="NAP Robustness Verifier for a Given Label")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--nap_file', type=str, default="mined_naps_small_net.json", help='NAP file (JSON).')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--bab', action='store_true', help='Use BaB for verification.')
    parser.add_argument('--gurobi_p', type=int, default=1, help='Number of threads for Gurobi .') 
    parser.add_argument('--timeout', type=int, default=3000, help='Timeout .') 
    parser.add_argument('--epsilon', type=float, default=1, help='Radius for input perturbation.')


    args = parser.parse_args()
    label=args.label
    nap_file=args.nap_file
    with open(nap_file, "r") as f:
            naps = json.load(f)
    refined_nap = get_label_nap(naps, label)
    epsilon=1

    
    verifier= Verifier(args.model, args.json, False,args.timeout)
   
    print(f"[Info ]Before minimizing the Nap specification for label{label} looks like  ")
    display_nap_array(refined_nap) 


    shortened_nap=shorten_Nap(refined_nap,epsilon, label,simple_order_neurons, verifier)

 
