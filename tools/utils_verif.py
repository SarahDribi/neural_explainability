import argparse
import torch
import json
import copy
from plnn.proxlp_solver.propagation import Propagation
from tools.bab_tools.model_utils import one_vs_all_from_model
from tools.bab_tools import vnnlib_utils
from tools.bab_tools.bab_runner import bab_from_json, bab_output_from_return_dict
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
import plnn.branch_and_bound.utils as bab_utils
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import time
"""
I run this file like this from a repo ahead python3 tools/nap_robustness_from_onnx.py   --model tools/mnist-net_256x4.onnx 
  --label 8   --nap_file tools/mined_naps.json   --json bab_configs/mnistfc_vnncomp21.json


"""

def get_label_picture(label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    for x, y in dataset:
        if y==label:
            return x


def get_label_nap(mined_Naps, label):
    nap = mined_Naps[label]
    print(f"[DEBUG] Using NAP for label {label}")
    print(f"[DEBUG] NAP has {len(nap)} relu hidden layers")
    for i, layer_nap in enumerate(nap):
        print(f"[DEBUG]   NAP relu Layer {i} has = {len(layer_nap)} activations status")
    return nap


def update_bounds_with_nap(ubs, lbs, nap, label):
    print(f"[DEBUG] Total bound layers: {len(ubs)} (includes input and output)")
    for i in range(1, len(ubs) - 1):  # skip input layer to start applying Nap update
        nap_layer_index = i - 1
        if nap_layer_index >= len(nap): # if the nap array len is reached 
            #it means that we are on the  output / extra verification layers
            print(f"NAP update only for inner bounds, skipping because this is either an  output or/an extra verif layer.")
            continue

        layer_nap = nap[nap_layer_index]
        num_neurons = ubs[i].shape[-1]
        print(f"[DEBUG] Intermidiate Layer {i}: NAP length = {len(layer_nap)}, Bounds neurons = {num_neurons}")

        for j in range(min(len(layer_nap), num_neurons)):
            print(f"[DEBUG] Now working on the {j} th neuron of {i} th relu layer ")
            if layer_nap[j] == 1:
                print(f"    [ACTive neuron in Nap] Layer {i}, Neuron {j}")
                lbs[i][0][j] = torch.max(torch.tensor(0.0, device=lbs[i].device), lbs[i][0][j] )
            elif layer_nap[j] == 0:
                print(f"    [Inactive neuron in Nap] Layer {i}, Neuron {j}")
                ubs[i][0][j] = torch.min( ubs[i][0][j],torch.tensor(0.0,device=ubs[i].device))
                   
        

def sanity_check(original_lbs, updated_lbs, original_ubs, updated_ubs,num_relu_layers):
    # ensure that the output layer bounds are tigher
    #the output layer comes directly after last relu layer
    final_layer = num_relu_layers+1
    tighter = True
    for j in range(len(original_lbs[final_layer][0])):
        if updated_lbs[final_layer][0][j] < original_lbs[final_layer][0][j]:
            print(f"[WARN not normal] Lower bound got looser after adding Nap constraint {j}")
            tighter = False
        if updated_ubs[final_layer][0][j] > original_ubs[final_layer][0][j]:
            print(f"[WARN not normal] Upper bound got looser after adding Nap constraint {j}")
            tighter = False
    return tighter



def compute_bounds_with_nap_around_input(model_path,input,epsilon, label, nap, use_gpu):
    print(f"[INFO] Loading ONNX model from {model_path}")
    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct, "ONNX model conversion mismatch."
    assert vnnlib_utils.is_supported_model(model), "Model structure unsupported."

        # Load one example image from the dataset with the correct label
    center_img = input
    input_point = center_img.to(torch.float32)
   
    input_bounds = torch.stack([(input_point - epsilon).clamp(0, 1), (input_point + epsilon).clamp(0, 1)], dim=-1)
    if epsilon==-1: #whole input space
         input_bounds = torch.stack([torch.zeros_like(input_point), torch.ones_like(input_point)], dim=-1)


    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype)

    net = one_vs_all_from_model(
        torch.nn.Sequential(*layers),
        label,
        domain=input_bounds,
        use_ib=True,
        gpu=use_gpu,
    )

    domain_batch = input_bounds.unsqueeze(0)
    if use_gpu:
        net = [layer.cuda() for layer in net]
        domain_batch = domain_batch.cuda()

    prop = Propagation(net, type="best_prop", params={"best_among": ["KW", "crown"]})
    with torch.no_grad():
        prop.define_linear_approximation(domain_batch)

    lbs = prop.lower_bounds
    ubs = prop.upper_bounds

    print("\n[DEBUG] Layer-wise bound shapes BEFORE NAP:")
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        print(f"  Layer {i}: LB shape = {lb.shape}, UB shape = {ub.shape}")

    lbs_orig = copy.deepcopy(lbs)
    ubs_orig = copy.deepcopy(ubs)


    update_bounds_with_nap(ubs, lbs, nap, label)

    print("\n[DEBUG] Layer-wise bounds AFTER applying NAP:")
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        print(f"  Layer {i}: LB min={lb.min():.4f}, max={lb.max():.4f} | UB min={ub.min():.4f}, max={ub.max():.4f}")

    if sanity_check(lbs_orig, lbs, ubs_orig, ubs,len(nap)):
        print(" output Bounds are tighter after applying NAP.")
    else:
        print(" Warning: Bounds got losser check the code !")

    domain_nap = torch.stack([lbs[0][0], ubs[0][0]], dim=-1)
    return net, domain_nap.unsqueeze(0),lbs,ubs
def get_adversarial_example():
    return

def verify_nap_property_around_input(model_path,input,epsilon, json_config, nap, label, use_gpu,timeout):
    
    
    layers, domain,lbs,ubs = compute_bounds_with_nap_around_input(model_path,input,epsilon, label, nap, use_gpu)
    bab=False


    if bab:
        if json_config:
            with open(json_config, "r") as f:
                config = json.load(f)
        else:
            print("[INFO] Using default BaB config")
            config = {
        "batch_size": 2000,
        "initial_max_domains": 500,
        "decision_thresh": 0.1,
        "score_function": "kfsb",
        "sort_domain_interval": 5,
        "max_domains": 10000,
        "branching": "relu_heuristic",
        "bound_prop_method": {
            "root": {
                "best": {}
            }
        },
        "cut": False
    }


        return_dict = {}
        print("\n[INFO] Running BaB...")
        bab_from_json(config, layers, domain, return_dict,
                    nn_name="mnist-nap", instance_timeout=500, gpu=use_gpu,precomputed_ibs=(lbs,ubs))
        del config  # prevent reuse issues

        result, nodes = bab_output_from_return_dict(return_dict)
        print(f"\n[RESULT] BaB verification status: {result}")
        print(f"[INFO] BaB visited {nodes} nodes")
    else:
        n_threads=3 # 3 threads par defaut
        anderson_mip_net = AndersonLinearizedNetwork(
                layers, mode="mip-exact", decision_boundary=0.0)
        #  add a flag of usingNap to TRUE or False as a class attribute
        #  
        """
         prop_params = {
        'nb_steps': 5,
        'initial_step_size': 1,
        'step_size_decay': 0.98,
        'betas': (0.9, 0.999),
        }
        
        prop_net = Propagation(layers, type='alpha-crown', params=prop_params)
        prop_net.build_model_using_bounds(domain.unsqueeze(0), (lbs, ubs))
        lb= prop_net.compute_lower_bound(node=(-1, 0))
        print(lb)
        """
       

        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
            domain.unsqueeze(0), lbs, ubs, squeeze=True)
        anderson_mip_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                    n_threads=3)

        sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=timeout, insert_cuts=False)
        #check  if sat_satus is None and global_lb is infinity 
        # Mark this as timeout 
        #import pdb; pdb.set_trace()
        return sat_status
    

def compute_bounds_around_input(model_path,input,epsilon, label, use_gpu):
    print(f"[INFO] Loading ONNX model from {model_path}")
    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct, "ONNX model conversion mismatch."
    assert vnnlib_utils.is_supported_model(model), "Model structure unsupported."

        # Load one example image from the dataset with the correct label
    center_img = input
    input_point = center_img.to(torch.float32)
   
    input_bounds = torch.stack([(input_point - epsilon).clamp(0, 1), (input_point + epsilon).clamp(0, 1)], dim=-1)
    if epsilon==-1: #whole input space
         input_bounds = torch.stack([torch.zeros_like(input_point), torch.ones_like(input_point)], dim=-1)


    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype)

    net = one_vs_all_from_model(
        torch.nn.Sequential(*layers),
        label,
        domain=input_bounds,
        use_ib=True,
        gpu=use_gpu,
    )

    domain_batch = input_bounds.unsqueeze(0)
    if use_gpu:
        net = [layer.cuda() for layer in net]
        domain_batch = domain_batch.cuda()

    prop = Propagation(net, type="best_prop", params={"best_among": ["KW", "crown"]})
    with torch.no_grad():
        prop.define_linear_approximation(domain_batch)

    lbs = prop.lower_bounds
    ubs = prop.upper_bounds

    

    domain_nap = torch.stack([lbs[0][0], ubs[0][0]], dim=-1)
    return net, domain_nap.unsqueeze(0),lbs,ubs

def verify_robustness_around_input(model_path,input,epsilon, json_config, label, use_gpu,timeout):
    layers, domain,lbs,ubs = compute_bounds_around_input(model_path,input,epsilon, label, use_gpu)
    bab=False


    if bab:
        if json_config:
            with open(json_config, "r") as f:
                config = json.load(f)
        else:
            print("[INFO] Using default BaB config")
            config = {
        "batch_size": 2000,
        "initial_max_domains": 500,
        "decision_thresh": 0.1,
        "score_function": "kfsb",
        "sort_domain_interval": 5,
        "max_domains": 10000,
        "branching": "relu_heuristic",
        "bound_prop_method": {
            "root": {
                "best": {}
            }
        },
        "cut": False
    }


        return_dict = {}
        print("\n[INFO] Running BaB...")
        bab_from_json(config, layers, domain, return_dict,
                    nn_name="mnist-nap", instance_timeout=500, gpu=use_gpu,precomputed_ibs=(lbs,ubs))
        del config  # prevent reuse issues

        result, nodes = bab_output_from_return_dict(return_dict)
        print(f"\n[RESULT] BaB verification status: {result}")
        print(f"[INFO] BaB visited {nodes} nodes")
    else:
        n_threads=3 # 3 threads par defaut
        anderson_mip_net = AndersonLinearizedNetwork(
                layers, mode="mip-exact", decision_boundary=0.0)
        #  add a flag of usingNap to TRUE or False as a class attribute
        #  add this FLAG to the vlass
        """
        add a propaga
       
        """
        
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
            domain.unsqueeze(0), lbs, ubs, squeeze=True)
        anderson_mip_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                    n_threads=3)

        sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=timeout, insert_cuts=False)
        #import pdb; pdb.set_trace()
        return sat_status


    




if __name__ == '__main__':
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

    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(args.model)
    print("\n[INFO] Model structure to check that we are alr:")
    print(model)

    torch.manual_seed(0)
    #global_check(args)
    print(f"Verifying the NAp {args.label}'s robustness ...............")
    
