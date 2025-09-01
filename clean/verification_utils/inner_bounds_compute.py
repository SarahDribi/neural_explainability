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
import time
from clean.verification_utils.nap_inner_bounds_enforcing import update_bounds_with_nap
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "oval-bab", "tools", "bab_tools"))
from tools.bab_tools import vnnlib_utils


    


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

    

    
    return net, domain_batch,lbs,ubs

   






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
        gpu=use_gpu, # I should add a variable called num_classes here that changes with data
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

    


    return net,domain_batch,lbs,ubs
