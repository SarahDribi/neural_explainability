





from clean.verification_utils.inner_bounds_compute import compute_bounds_with_nap_around_input




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



# I am going to add a boolean flag to see if the decision was made ealier or not
def verify_nap_property_around_input(model_path,input,epsilon, json_config, nap, label, use_gpu,timeout,num_classes=10):
    
    
    layers, domain,lbs,ubs = compute_bounds_with_nap_around_input(model_path,input,epsilon, label, nap, use_gpu,num_classes=num_classes)
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
        del config  # IN ORDER TO prevent reuse issues
        status, nodes = bab_output_from_return_dict(return_dict)
        #if status == "False":    
        #    is_safe = True;  timed_out = False
            
        #elif status == "True":     
        #    is_safe = False; timed_out = False
        #elif status in {"timeout", "ET"}:
        #   is_safe = False; timed_out = True
        #else:  
        #   is_safe = False; timed_out = True
        #if is_safe:
        #return True,False
    # if false alarm or timeout then use the mip

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
        if sat_status is None :
            #print("[INFO] Timeout reached, returning None")
            return False,True  # robustness is not verified within reached truley reached timeout 
        return not sat_status,False  # decision was  made before not reached time limit