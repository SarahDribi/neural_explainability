
# first find a handful of images in  a label stisfying the exclusive nap robustness on epsilon (robust on epsimon with nap but not without)
from nap_coarsening import (shorten_nap_around_input, stochasticShorten,shorten_nap_around_input_complex_heuristic)
from utils import (diff_naps,get_coarsening_percentage,   load_model)

from ordering_strategies import (simple_order_neurons,get_shuffled_neurons,region_insensitivity_order) # I already defined them in this file
from optimal_region_coarsening import(
    get_nap_specification_exclusive   # this function finds the input/inputs that are exclusively nap_augmented robust at a label 
)
from verifier import Verifier
import argparse
import torch
# I am only importing them




def run_script(args):
    # I need to define what to do here 
    # call it for one label this time 
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)

    
    model = load_model(args.model)
    label=args.label

    process_nap_exclusive_example(label=label,verifier=verifier,model=model,args=args,device="cpu")


def process_nap_exclusive_example(label, verifier, model, args, device):
    image, epsilon, nap = get_nap_specification_exclusive(
        label=label,
        verifier=verifier,
        model=model,
        args=args,
        device=device,
        delta=0.003,  # I can change it afterwards
        return_all=False
    )
    
    print(f"[INFO] Found NAP-exclusive robust input for label {label} at epsilon={epsilon:.4f}")
    
    # Applying different heuristics
    simple_heuristics =[]# [simple_order_neurons,get_shuffled_neurons]#,get_shuffled_neurons,get_shuffled_neurons,get_shuffled_neurons]
    simple_heur_names = []#"simple", "first_random"]#,"second_random","third_random","fourth_random"]
    #advanced_heuristics=[region_insensitivity_order]
    #advanced_heur_names=["region_insensitivity_order"]

    # I d better use a class called heuristic that comes with a name and a function
    coarsened_results = []

    for i, heuristic in enumerate(simple_heuristics):
        coarsened = shorten_nap_around_input(nap, image, label, epsilon, heuristic, verifier)
        diff = diff_naps(nap, coarsened)
        percent = get_coarsening_percentage(nap, coarsened)
       
        coarsened_results.append((simple_heur_names[i], coarsened, diff, percent))
    """
    for i, heuristic in enumerate(advanced_heuristics):
        coarsened = shorten_nap_around_input_complex_heuristic(nap,args, image, label, epsilon, heuristic, verifier)
        diff = diff_naps(nap, coarsened)
        percent = get_coarsening_percentage(nap, coarsened)
       
        coarsened_results.append((advanced_heur_names[i], coarsened, diff, percent))
    """
    

    # Stochastic trials
    stochastic_trials = []
    for i in range(1):
        coarsened,successful_iterations = stochasticShorten(nap, image, label, epsilon, verifier,0.1)
        diff = diff_naps(nap, coarsened)
        percent = get_coarsening_percentage(nap, coarsened)
        print(f"[Stochastic #{i}] Coarsened neurons: {diff} | Remaining %: {percent:.2f}| Coarsened {coarsened} | succesful_iterations{successful_iterations}")
        stochastic_trials.append((coarsened, diff, percent,successful_iterations))
    # For console clear display i ll add additional loops 
    for i in range(len(coarsened_results)):
        heur_name, coarsened_nap, diff, percent=coarsened_results[i]
        print(f"[{heur_name}] Coarsened neurons: {diff} | Remaining %: {percent:.2f} | Nap coarsened {coarsened_nap}")
    for i in range(0):
        coarsened, diff, percent=stochastic_trials[i]
        print(f"[Stochastic #{i}] Coarsened neurons: {diff} | Remaining %: {percent:.2f} | Nap coarsened {coarsened}")
        


    
    #best = max(coarsened_results, key=lambda x: x[2])  # max coarsening
    #print(f"\n[SUMMARY] Best deterministic coarsening: {best[0]} with {best[2]} neurons removed ({best[3]:.2f}% kept)")

    return {
        "label": label,
        "epsilon": epsilon,
        
    }

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="NAP Coarsening Process")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--timeout', type=int, default=700, help='Timeout in seconds.')
    parser.add_argument('--theta', type=float, default=0.2, help='Probability of abstracting each neuron during stochastic coarsening.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of stochastic coarsening iterations.')
 
    
    args = parser.parse_args()
   
    model = load_model(args.model)
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
    
    image, epsilon, nap = get_nap_specification_exclusive(
        label=args.label,
        verifier=verifier,
        model=model,
        args=args,
        device="cpu",
        delta=0.003,  # I can change it afterwards
        return_all=False
    )
    
    
    #run_script(args=args)
    
    print(f"[INFO] Found NAP-exclusive robust input for label {args.label} at epsilon={epsilon:.4f}")
    print(f"nap is {nap}")
    