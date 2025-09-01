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




def compute_bounds(model_path, label, nap_file, use_gpu,eps,use_Nap=False):
    print(f"[INFO] Loading ONNX model from {model_path}")
    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct, "ONNX model conversion mismatch."
    assert vnnlib_utils.is_supported_model(model), "Model structure unsupported."

        # Load one example image from the dataset with the correct label
    center_img = get_label_picture(label)  # shape: (784,)
    input_point = center_img.to(torch.float32)
    #if eps==-1 this means we are verifiying Nap robustness
    #property witch means we are checking entire input space
    input_bounds = torch.stack([(input_point - eps).clamp(0, 1), (input_point + eps).clamp(0, 1)], dim=-1)

    if eps==-1:
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


def adversarial_example(model_path, json_config, nap_file,sample, use_gpu,eps,nap_augmented=False):
    layers, domain,lbs,ubs = compute_bounds(model_path, label, nap_file, use_gpu,eps,nap_augmented)


    if args.bab:
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
        anderson_mip_net = AndersonLinearizedNetwork(
                layers, mode="mip-exact", decision_boundary=0.0)

        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
            domain.unsqueeze(0), lbs, ubs, squeeze=True)
        anderson_mip_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                    n_threads=args.gurobi_p)

        sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=args.timeout, insert_cuts=False)
        #import pdb; pdb.set_trace()
        #if an adversarial attack is found then return the corresponding input 
        #that made it 
        #we can find adversarial attack return it
        #else return None 
        
        return None

  


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
    

