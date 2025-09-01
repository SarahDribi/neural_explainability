from tqdm import tqdm 
import copy

from clean.heuristics.model_utils import get_neuron_ordering

"""
This function takes  a nap (see the nap format in nap extraction file) and returns a relaxed nap
meaning a nap where the constraint on the activation of the passed neuron is relaxed
neuron is reffered to by its layer index and index at that given layer 
"""
def make_abstract(nap, neuron):
    layer_idx, idx = neuron
    nap_copy = copy.deepcopy(nap)
    nap_copy[layer_idx][idx] = -1
    return nap_copy

"""
This function gives the natural ordering of neurons using their nap

"""
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
This function follows this approach :
-goes given a neuron ordering :
- selecting a neuron :
- trying to abstract it 
-if the region is no longer robust using this relaxed contraint on that neuron activation
-nothing is done
-else it s updated

"""
def shorten_nap_around_input(nap, input, label, epsilon, order_neurons_func, verifier):
    nap_copy = copy.deepcopy(nap)
    timeout_flags = [[False for _ in layer] for layer in nap_copy]
    new_eps = epsilon
    verification_result = verifier.is_verified_nap(nap_copy, input, label, new_eps)
    if not verification_result:
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap_copy)

    # Progress bar for the coarsening loop
    for neuron in tqdm(order_neurons_func(neurons), desc="Coarsening NAP", unit="neuron"):
        a, b = neuron
        modified_nap = make_abstract(nap_copy, neuron)
        verification_result ,timed_out= verifier.is_verified_nap_small_timeout(modified_nap,input, label, epsilon)
        if verification_result:
            nap_copy = modified_nap
            print(f"[Coarsening] Neuron ({a},{b}) coarsened successfully.")
        if timed_out:
            timeout_flags[a][b] = True
            print(f"[Coarsening ]Neuron ({a},{b}) timed out during verification.")

    return nap_copy,timeout_flags



"""
Same , but what changes is using a wise heuristic 
- given what we can assume about the neurons in tat region 




"""

"""
def shorten_nap_around_input_complex_heuristic(nap,args,input, label, epsilon, order_neurons_func, verifier): # like region_insensivity_order(neurons, model_tracked, x_nap, x_input, epsilon, max_samples=100, noise_factor=1.0):
    nap_copy=copy.deepcopy(nap)
    timeout_flags = [[False for _ in layer] for layer in nap_copy]

    new_eps=epsilon
    verification_result = verifier.is_verified_nap(nap_copy, input, label, new_eps)
    if not verification_result:
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap_copy)
    count=0
    for neuron in order_neurons_func(neurons,args.model,nap,input,epsilon):
        
        a,b=neuron
        modified_nap = make_abstract(nap_copy, neuron)
        verification_result ,timed_out= verifier.is_verified_nap_small_timeout(modified_nap,input, label, epsilon)
        if verification_result:
            nap_copy = modified_nap
            print(f"[Coarsening ]Coarsening Neuron ({a}{b}) worked")
        if timed_out:
            timeout_flags[a][b] = True
            print(f"[Coarsening ]Neuron ({a},{b}) timed out during verification.")

    return nap_copy,timeout_flags


"""

def shorten_nap_around_input_complex_heuristic(
    nap,
    args,
    input,
    label,
    epsilon,
    order_neurons_func,
    verifier,
):
    """
    Version “complexe”: on passe des kwargs riches à l’heuristique (ex: modèle ONNX, x_nap, epsilon…)
    Convention: l’heuristique doit accepter **kwargs et ignorer ce dont elle n’a pas besoin.
    """
    
    nap_copy = copy.deepcopy(nap)
    timeout_flags = [[False for _ in layer] for layer in nap_copy]
    new_eps = epsilon
    verification_result = verifier.is_verified_nap(nap_copy, input, label, new_eps)
    if not verification_result:
        print("[Info] NAP is not robust initially.")
        return None
    x_ref_nap = nap

    order_kwargs = dict(
        model_onnx=getattr(args, "model", None),
        x_nap=x_ref_nap,
        x_input=input,
        epsilon=epsilon,
        
    )
    neurons = get_neurons(nap)
    # the heuristic ordering 
    ordered = order_neurons_func(neurons, **order_kwargs)
     # Progress bar for the coarsening loop
    for neuron in tqdm(order_neurons_func(neurons), desc="Coarsening NAP", unit="neuron"):
        a, b = neuron
        modified_nap = make_abstract(nap_copy, neuron)
        verification_result ,timed_out= verifier.is_verified_nap_small_timeout(modified_nap,input, label, epsilon)
        if verification_result:
            nap_copy = modified_nap
            print(f"[Coarsening] Neuron ({a},{b}) coarsened successfully.")
        if timed_out:
            timeout_flags[a][b] = True
            print(f"[Coarsening ]Neuron ({a},{b}) timed out during verification.")

    return nap_copy,timeout_flags




def coarsen_heuristic(nap, input, label, epsilon,heuristic_name, verifier,model_path):
    nap_copy = copy.deepcopy(nap)
    timeout_flags = [[False for _ in layer] for layer in nap_copy]
    new_eps = epsilon
    verification_result = verifier.is_verified_nap(nap_copy, input, label, new_eps)
    if not verification_result:
        print("[Info] NAP is not robust initially.")
        return None

    neurons = get_neurons(nap_copy)
    # order them using the heuristic name
    ordered= get_neuron_ordering(heuristic_name, {
        'model_onnx': model_path,
        'x_nap': nap_copy,
        'x_input': input,
        'epsilon': epsilon,
        'x_label': label
    })
    # Progress bar for the coarsening loop
    for neuron in tqdm(ordered, desc="Coarsening NAP", unit="neuron"):
        a, b = neuron
        modified_nap = make_abstract(nap_copy, neuron)
        verification_result ,timed_out= verifier.is_verified_nap_small_timeout(modified_nap,input, label, epsilon)
        if verification_result:
            nap_copy = modified_nap
            print(f"[Coarsening] Neuron ({a},{b}) coarsened successfully.")
        if timed_out:
            timeout_flags[a][b] = True
            print(f"[Coarsening ]Neuron ({a},{b}) timed out during verification.")

    return nap_copy,timeout_flags



    


