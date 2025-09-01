import argparse
import torch
import copy
import json
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

from tools.bab_tools.model_utils import one_vs_all_from_model
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.bab_tools.bab_runner import bab_from_json, bab_output_from_return_dict


def find_mnist_image_with_label(target_label):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    for img, label in dataset:
        if label == target_label:
            return img.unsqueeze(0), label
    raise ValueError(f"No MNIST image with label {target_label} found.")


def parse_input():
    torch.manual_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ONNX model path')
    parser.add_argument('--label', type=int, required=True, help='Target label to verify')
    parser.add_argument('--json', type=str, default="bab_configs/mnistfc_vnncomp21.json", help='OVAL BaB json config')
    parser.add_argument('--gpu', action='store_true', help="Run BaB on GPU")
    parser.add_argument('--eps', type=float, default=0.065, help='L_inf radius')
    args = parser.parse_args()

    print(f"[INFO] Loading ONNX model from {args.model}")
    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.model)
    assert model_correctness, "Model conversion failed!"
    assert vnnlib_utils.is_supported_model(model), "Unsupported model structure."

    input_point, y = find_mnist_image_with_label(args.label)

    input_bounds = torch.stack([
        (input_point - args.eps).clamp(0, 1),
        (input_point + args.eps).clamp(0, 1)
    ], dim=-1)

    with torch.no_grad():
        layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype=dtype)

    verif_layers = one_vs_all_from_model(
        torch.nn.Sequential(*layers), y, domain=input_bounds,
        max_solver_batch=1000, use_ib=True, gpu=args.gpu
    )

    return verif_layers, input_bounds, args, layers  # added `layers`

def run_bab_from_json():
    layers, domain, args, raw_layers = parse_input()
    timeout = 300

    with open(args.json) as json_file:
        json_params = json.load(json_file)

    return_dict = dict()
    bab_from_json(json_params, layers, domain, return_dict, None, instance_timeout=timeout, gpu=args.gpu)
    del json_params

    bab_out, bab_nb_states = bab_output_from_return_dict(return_dict)
    print(f"[RESULT] BaB output state: {bab_out}, number of visited nodes: {bab_nb_states}")

    # Debug: run the model on the center point and print logits
    with torch.no_grad():
        model = torch.nn.Sequential(*raw_layers)

        # Use the center point of the L∞ ball
        x = ((domain[..., 0] + domain[..., 1]) / 2).unsqueeze(0)
        logits = model(x)

        predicted_class = torch.argmax(logits, dim=1).item()
        max_other = torch.max(torch.cat([logits[0, :args.label], logits[0, args.label+1:]]))
        margin = logits[0, args.label] - max_other

        print(f"[DEBUG] Logits: {logits.squeeze().tolist()}")
        print(f"[DEBUG] Predicted class: {predicted_class}")
        print(f"[DEBUG] True label: {args.label}")
        print(f"[DEBUG] Margin (logit[{args.label}] - max_other): {margin.item()}")

if __name__ == '__main__':
    run_bab_from_json()
