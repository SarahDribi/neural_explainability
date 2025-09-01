import os
import sys
from collections import defaultdict
from typing import List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import json
import copy
import numpy as np

import argparse
# OVAL-BaB path
# Always ensure the project root (modified-oval-bab/) is in PYTHONPATH
from clean.dataset.other_dataset import get_label_input
from tools.bab_tools import vnnlib_utils
from clean.nap_extraction.nap_utils import  display_nap_array
from clean.verifier.verifier_base import Verifier

"""

    This is a module that contructs a model where activations can be tracked 
from an onnx model given its path 
    The nap extraction function takes a model , an input tensor and returns 
the activation pattern as follows :

Example:


output


"""


def load_model(onnx_path: str) -> nn.Module:
    print("Loading ONNX model...")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()


class TrackedActivationsModel(nn.Module):
    def __init__(self, base: nn.Sequential, relu_layers: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.base = base
        self.relu_layers = relu_layers
        self.activations = {}

    def forward(self, x):
        self.activations = {}
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.relu_layers:
                self.activations[f"relu_{i}"] = x.clone()
        return x



def nap_extraction(model: TrackedActivationsModel, x: torch.Tensor) -> List[List[int]]:
    _ = model(x.unsqueeze(0))
    nap = []
    for layer_name in sorted(model.activations.keys()):
        activations = model.activations[layer_name].flatten()
        layer_nap = [1 if val.item() > 0 else 0 for val in activations]
        nap.append(layer_nap)
    return nap



def nap_extraction_from_onnx(onnx_model_path,x):
    base_model = load_model(onnx_model_path)
    model_tracked = TrackedActivationsModel(base_model)
    nap=nap_extraction(model_tracked,x)
    return nap


def build_parser():
    p = argparse.ArgumentParser(description="Run heuristic tests on NAP coarsening.")
    p.add_argument('--model', type=str, required=True, help='Path to the ONNX model.')
    p.add_argument('--json', type=str, default=None, help='Optional JSON spec/config for the verifier.')
    p.add_argument('--gpu', type=int, default=-1, help='GPU id to use, -1 for CPU.')
    p.add_argument('--timeout', type=int, default=120, help='Timeout (seconds) for verification calls.')

    # NAP spec arguments
    p.add_argument('--device', type=str, default='cpu', help='cpu | cuda')
    p.add_argument('--label', type=int, default=1, help='Target label for the NAP specification.')
    

    return p

# ADD THE PARSING ARGUMENTS IN THE PARSER
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    
    onnx_path = "artifacts/wbc_mlp.onnx"
    x = get_label_input(1) # is already scaled 
    print(x)
    nap=nap_extraction_from_onnx(onnx_path,x)
        # Create verifier 
    try:
        verifier = Verifier(onnx_path, args.json, args.gpu, args.timeout)
    except TypeError:
        verifier = Verifier(model=onnx_path, json=args.json, gpu=args.gpu, timeout=args.timeout)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Verifier: {e}", file=sys.stderr)
        sys.exit(1)
    # I need to know if the 
    res=verifier.is_verified_region(x,1,0)
    print((res))



