import os
import json
import torch
from collections import defaultdict
from verifier import Verifier
from utils import load_model
from optimal_region_coarsening import( find_max_epsilon_around_input,find_max_epsilon_around_input_without_nap)
from utils import (get_label_inputs,nap_extraction_from_onnx)
def get_nap_specification_exclusive(
    label: int,
    verifier,
    model,
    args,
    device: str = "cpu",
    delta: float = 0.001,
    return_all: bool = False,
    number=5
):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    inputs = get_label_inputs(label)  
    print(f"[INFO] Searching for NAP-exclusive robust inputs for label {label}")

    results = []
    count=0
    for i in range(len(inputs)):
        image_input = inputs[i].to(device).unsqueeze(0)

        
        prediction = model(image_input).argmax(dim=1).item()
        if prediction != label:
            continue

        
        nap = nap_extraction_from_onnx(args.model, image_input.squeeze(0))

        
        max_eps_nap = find_max_epsilon_around_input(
            image_input, nap, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )

        
        max_eps_base = find_max_epsilon_around_input_without_nap(
            image_input, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )


        if max_eps_nap > max_eps_base + delta:
            print(f" Found input where NAP improves robustness! at index {i} epsilons are  {max_eps_nap} and  {max_eps_base}")
            result = (image_input.squeeze(0).cpu(), max_eps_nap,nap )
            if not return_all:
                return result
            else:
                results.append(result)
                count+=1
                if count==number:
                    return results

    if return_all:
        return results
    else:
        print("No example found where NAP strictly improves robustness.")
        return None

   

def collect_nap_exclusive_for_label(label, verifier, model, args, delta=0.003):
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    model = model.to(device)

    print(f"[INFO] Getting all NAP-exclusive robust examples for label {label}")
    results = get_nap_specification_exclusive(
        label=label,
        verifier=verifier,
        model=model,
        args=args,
        device=device,
        delta=delta,
        number=3,
        return_all=True
    )

    examples = []
    

    if not results:
        print(f" No exclusive NAPs found for label {label}")
        return []

    for image_tensor, epsilon, nap in results:
        nap_tuple = tuple(nap.tolist()) if isinstance(nap, torch.Tensor) else tuple(nap)
        
        examples.append({
            "epsilon": float(epsilon),
            "nap": list(nap_tuple),
            "input": image_tensor.numpy().tolist()
        })
        if len(examples) == 7:
            break

    print(f" Collected {len(examples)} exclusive examples for label {label}")
    return examples


def collect_all_labels(args, output_dir="exclusive_data_per_label", delta=0.003):
    verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
    model = load_model(args.model)

    os.makedirs(output_dir, exist_ok=True)

    label=args.label
    examples = collect_nap_exclusive_for_label(label, verifier, model, args, delta)

    label_file = os.path.join(output_dir, f"label_{label}_exclusive.json")
    with open(label_file, "w") as f:
      json.dump(examples, f, indent=2)

    print(f"Saved {len(examples)} examples to {label_file}")


def load_exclusive_input(label: int, index: int, folder="exclusive_data_per_label"):
    path = f"{folder}/label_{label}_exclusive.json"
    with open(path, "r") as f:
        data = json.load(f)
    if index >= len(data):
        raise IndexError(f"Only {len(data)} examples found for label {label}, index {index} is out of range.")
    
    entry = data[index]
    image = torch.tensor(entry["input"], dtype=torch.float32)
    nap = torch.tensor(entry["nap"], dtype=torch.float32)
    epsilon = float(entry["epsilon"])
    return image.unsqueeze(0), nap, epsilon


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect NAP-exclusive robust examples per label.")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--timeout', type=int, default=700, help='Timeout in seconds.')
    parser.add_argument('--delta', type=float, default=0.003, help='Required robustness gap (NAP vs no-NAP).')
    parser.add_argument('--outdir', type=str, default="exclusive_data_per_label", help='Directory to save output JSON files.')

    args = parser.parse_args()
    collect_all_labels(args, output_dir=args.outdir, delta=args.delta)