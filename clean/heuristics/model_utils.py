"""
ALL the ingreDients for the heuristics module

"""



# this function takes a string # heuristic name and returns a neuron ordering accordingly



# I should take the model as a parameter 
# The input 
# sometimes the dateset is manipulated 

# args are the image , the model , the nap , the epsilon, the label
# args has the following keys:
# - model_onnx: the ONNX model
# - x_nap: the NAP of the input
# - x_input: the input tensor
# - epsilon: the epsilon for sampling
# - x_label: the label for the input
# get neuons function 
from clean.nap_extraction.extract_nap import nap_extraction_from_onnx

from clean.heuristics.simple_order_heur import simple_order_neurons
from clean.heuristics.random_order_heur import get_shuffled_neurons
from clean.heuristics.neighbor_sampling_heur import region_insensitivity_order
from clean.heuristics.preactivation_heur import higher_impact_order_in_memory 


def get_neurons(nap):
    neurons = []
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            neurons.append((i, j))
    return neurons
def get_neuron_ordering(heuristic_name,args):
    nap=args['x_nap']
    model_onnx=args['model_onnx']
    x_input=args['x_input']
    epsilon=args['epsilon']
    x_label=args['x_label']
    neurons= get_neurons(nap)
    if heuristic_name == "simple":
      
        return simple_order_neurons(neurons)
    elif heuristic_name == "region_sensitivity":
     
        return get_shuffled_neurons(neurons)
    elif heuristic_name == "preactivation_impact":
        ordered_neurons,_=higher_impact_order_in_memory(neurons,model_onnx,x_input,x_label)
        return ordered_neurons

    if heuristic_name == "random":
        return get_shuffled_neurons(neurons)
    if heuristic_name == "descending":
        return sorted(neurons, key=lambda x: x[0], reverse=True)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")
    
