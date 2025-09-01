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
from utils_verif import verify_robustness_around_input
from verifier import Verifier 

from utils import (
    load_model,
    TrackedActivationsModel,
    get_label_input,
    get_label_inputs,
    nap_extraction,
    nap_extraction_from_onnx,
    display_nap_array,
    diff_naps,
    get_coarsening_percentage


)
from nap_coarsening import(

shorten_nap_around_input,
stochasticShorten

)
# I have to make these names more descriptive
from optimal_region_coarsening import(
    find_max_epsilon_around_input,
    find_max_epsilon_around_input_without_nap,
    find_input_robust_at_epsilon,
    find_input_robust_at_epsilon_without_nap,
    get_nap_specification_exclusive   # this function finds the input/inputs that are robust at a label 
)







def save_nap_to_json(nap, filename):
    with open(filename, 'w') as f:
        json.dump(nap, f)
    print(f"[INFO] Coarsened NAP saved to {filename}")
#For now its the identity
def simple_order_neurons(neurons):
    return neurons
def layer_wise_order(neurons):
    
    layer_priority = [3,2,1,0] #prioritizes coarsening in this order
    
    return sorted(neurons, key=lambda n: (layer_priority.index(n[0]), n[1]))

def last_layers_first_order(neurons):
    # inverser l ordre 
    return neurons[:-1]
# Main Execution

def get_label_input(label,index=0):
    l=get_label_inputs(label)
    return l[index]


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


       


    

def get_shuffled_neurons(neurons):

    random.shuffle(neurons)
    return neurons

    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="NAP Coarsening Process")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--timeout', type=int, default=700, help='Timeout in seconds.')
    parser.add_argument('--nap_file', type=str, default="mined_naps_small_net.json", help='NAP file (JSON).')
    parser.add_argument('--theta', type=float, default=0.2, help='Probability of abstracting each neuron during stochastic coarsening.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of stochastic coarsening iterations.')
 

    args = parser.parse_args()
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)

    
    model = load_model(args.model)
    
    """
    
    epsilon=0.04# for 7 0.05 works  # 2236 is 0.07 epsilon robust for 2   0.04 for nine works well 
    index, image, nap = find_input_robust_at_epsilon(
    eps=epsilon,
    label=label,

    args=args,
    verifier=verifier,
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
    """

    # coarsen this nap
    # try different strategies
    

    # mettre un tout petit tableau pour comparer les different ordering strategies
    """
    image=get_label_input(label,index)
    
    
    print(f"The index  is {index}")
    #test_all_layer_abstraction(label,nap,image,epsilon)
    print(f"The index  is {index}")
    
    coarsened=shorten_nap_around_input(nap,image, label, epsilon,simple_order_neurons, verifier)
    coarsened_by_layer=shorten_nap_around_input(nap,image, label, epsilon,get_shuffled_neurons, verifier)
    diff = diff_naps(nap, coarsened)
    print(f"[INFO] Coarsened neurons number: {diff}")
    diff_layers = diff_naps(nap, coarsened_by_layer)
    print(f"[INFO] Coarsened neurons number by layer ordering: {diff_layers}")
    coarsening_percentage = get_coarsening_percentage(nap, coarsened)
    print(f"[INFO] Coarsening achieved simple ordering: {coarsening_percentage:.2f}% remaining neurons")
    coarsening_percentage_by_layer = get_coarsening_percentage(nap, coarsened_by_layer)
    print(f"[INFO] Coarsening achieved random ordering : {coarsening_percentage_by_layer:.2f}% remaining neurons")
    
    print(f"{display_nap_array(coarsened_by_layer)}")
    print(f"{display_nap_array(coarsened)}")
    
    low=0.001
    high=0.9
    
    max_epsilon=find_max_epsilon_around_input(image, nap, label, verifier, low=0.001, high=1, tol=0.0001, max_iter=500)
   
    max_epsilon_withoutNap=find_max_epsilon_around_input_without_nap(image, label, verifier, low=0.001
                                                                     , high=1, tol=0.0001, max_iter=300)
    print(f"The maximum found robust epsilon around label with Nap specification is {max_epsilon}")
    print(f"The maximum found robust  epsilon around label without Nap specification {max_epsilon_withoutNap}")
    print(f"{nap}")
    
    
"""

    # add this test case 
    # add this test case
    #print(f"{verifier.is_verified_nap(nap,image,label,max_epsilon_withoutNap)}")
    # I have a special case here where it s robust without the Nap specification at a certain epsilon 
    # And it s not robust for the same epsilon ball under Nap specification wich really does nt make 
    # sense and needs further debugging
    
    # verify that it s Nap robust
    label=7
    image,max_epsilon,nap=get_nap_specification_exclusive(
    label=label,
    verifier=verifier,
    model=model,
    args=args,
    device= "cpu",
    delta= 0.003, # bigger delta
    return_all= False
     )
    
    """
    
    coarsened=shorten_nap_around_input(nap,image, label, max_epsilon,simple_order_neurons, verifier)
    coarsened_by_layer=shorten_nap_around_input(nap,image, label, max_epsilon,get_shuffled_neurons, verifier)
    a=[]
    for i in range(10):
        coarsened_by=shorten_nap_around_input(nap,image, label, max_epsilon,get_shuffled_neurons, verifier)
        a.append(coarsened_by)
    for i in range(len(a)):
            diff = diff_naps(nap, a[i])
            print(f"[INFO] Coarsened neurons number for random ordering number {i} was: {diff}")
            print(f"{display_nap_array(a[i])}")

            
                

    diff = diff_naps(nap, coarsened)
    print(f"[INFO] Coarsened neurons number: {diff}")
    diff_layers = diff_naps(nap, coarsened_by_layer)
    print(f"[INFO] Coarsened neurons number by layer ordering: {diff_layers}")
    coarsening_percentage = get_coarsening_percentage(nap, coarsened)
    print(f"[INFO] Coarsening achieved simple ordering: {coarsening_percentage:.2f}% remaining neurons")
    coarsening_percentage_by_layer = get_coarsening_percentage(nap, coarsened_by_layer)
    print(f"[INFO] Coarsening achieved random ordering : {coarsening_percentage_by_layer:.2f}% remaining neurons")

    
    print(f"{display_nap_array(coarsened_by_layer)}")
    print(f"{display_nap_array(coarsened)}")
    
    """
    coarsened,_=stochasticShorten(nap,image, label, max_epsilon,verifier,args.theta,args.iterations)

    diff = diff_naps(nap, coarsened)
    print(f"[INFO] Coarsened neurons number: {diff}")
    coarsening_percentage = get_coarsening_percentage(nap, coarsened)
    print(f"[INFO] Coarsening achieved simple ordering: {coarsening_percentage:.2f}% remaining neurons")

    
 # I also noticed that the bigger the delta between the both , the more diverse 
 #the explanation could be
    
    print(f"[INFO] Coarsened NAP after stochastic coarsening: {coarsened}")
    print(f"[INFO] NAP before stochastic coarsening: {nap}")
   
    
    
     
    
    
    
    
  
    
    
    
    
    
    
    
    
    
   
    
    
    
  
    
  
   
    
    
    
   



        
    
"""
def test_all_layer_abstraction(label,nap,input_image,epsilon):
    

    
    
    for i in range(len(nap)):
        for j in range(len(nap[i])):
                    nap[i][j]=-1

    label=label
    is_verified = verifier.is_verified_nap(nap, input_image, label,epsilon)

    assert is_verified, "Verification with evrything  abstracted success."
    print("[PASS] Verification with with all  abstracted  layers .")
"This function finds the maximum region for wich the nap holds as a spec"


"""
    
   
     


     
    
    
    
  
