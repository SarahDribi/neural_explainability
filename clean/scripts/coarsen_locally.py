#!/usr/bin/env python3
# scripts/coarsen_from_selected.py
import argparse
import json
import os
import sys
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Ensure project root importable
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project imports ---
from clean.coarsening.nap_coarsen import shorten_nap_around_input
from clean.coarsening.stochastic_approach import stochasticShorten
from clean.nap_extraction.nap_utils import diff_naps, get_coarsening_percentage
from clean.nap_extraction.extract_nap import load_model
from clean.heuristics.simple_order_heur import simple_order_neurons
from clean.heuristics.random_order_heur import get_shuffled_neurons
# from clean.heuristics.random_order_heur import get_shuffled_neurons  # optional
from clean.verifier.verifier_base import Verifier

# --- Torch is required (we load image.pt) ---
try:
    import torch
except Exception:
    torch = None


def load_selected_run(selected_dir: str, device: str, fallback_label=None, fallback_epsilon=None):
    """
    Load image.pt (tensor), nap.json and optionally results.json from selected_dir.
    Returns: (image_t: torch.Tensor [N,C,H,W], nap, label:int, epsilon:float)
    """
    image_path = os.path.join(selected_dir, "image.pt")
    nap_path = os.path.join(selected_dir, "nap.json")
    results_path = os.path.join(selected_dir, "results.json")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing image.pt in {selected_dir}")
    if not os.path.isfile(nap_path):
        raise FileNotFoundError(f"Missing nap.json in {selected_dir}")
    if torch is None:
        raise RuntimeError("PyTorch is required to load image.pt but is not available.")

    image_t = torch.load(image_path, map_location=device)

    with open(nap_path, "r", encoding="utf-8") as f:
        nap = json.load(f)

    label = fallback_label
    epsilon = fallback_epsilon

    if os.path.isfile(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                R = json.load(f)
            if label is None and "label" in R:
                label = int(R["label"])
            if epsilon is None and "epsilon" in R:
                epsilon = float(R["epsilon"])
        except Exception:
            pass

    if label is None:
        raise ValueError("Label not provided. Supply --label or ensure results.json contains 'label'.")
    if epsilon is None:
        raise ValueError("Epsilon not provided. Supply --epsilon or ensure results.json contains 'epsilon'.")

    # Ensure batch dim: (N,C,H,W)
    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0)
    elif image_t.ndim == 2:
        image_t = image_t.unsqueeze(0).unsqueeze(0)

    return image_t.to(device), nap, int(label), float(epsilon)


def process_loaded_example(image_t, nap, label, epsilon, verifier, args):
    """
    Run both deterministic and stochastic coarsening strategies.
    Returns a dict with detailed results.
    """
    # Deterministic heuristics (add more if needed)
    simple_heuristics = [simple_order_neurons,get_shuffled_neurons,get_shuffled_neurons]
    simple_heur_names = ["simple","shuffled1","shuffled2","shuffled3",]

    coarsened_results = []
    for name, heuristic in zip(simple_heur_names, simple_heuristics):
        coarsened, timeout_flags = shorten_nap_around_input(
            nap, image_t, label, epsilon, heuristic, verifier
        )
        diff = diff_naps(nap, coarsened)
        percent = get_coarsening_percentage(nap, coarsened)
        coarsened_results.append((name, coarsened, timeout_flags, diff, percent))

    # Stochastic variant
    coarsened_s, successful_iterations = stochasticShorten(
        nap, image_t, label, epsilon, verifier, args.theta, args.iterations
    )
    diff_val = diff_naps(nap, coarsened_s)
    percent_val = get_coarsening_percentage(nap, coarsened_s)
    print(
        f"[Stochastic] Coarsened neurons: {diff_val} | Remaining %: {percent_val:.2f} "
        f"| successful_iterations={successful_iterations}"
    )

    print(f"[INFO] Coarsened NAP (stochastic): {coarsened_s}")
    print(f"[INFO] Original NAP:               {nap}")

    return {
        "label": int(label),
        "epsilon": float(epsilon),
        "stochastic_diff": int(diff_val),
        "stochastic_remaining_percent": float(percent_val),
        "stochastic_successful_iterations": int(successful_iterations),
        "coarsened_nap": coarsened_s,
        "original_nap": nap,
        "coarsened_results": [
            {
                "heuristic": name,
                "coarsened_nap": coarsened_nap,
                "timeout_flags": timeout_flags,
                "diff": diff,
                "remaining_percent": percent,
            }
            for name, coarsened_nap, timeout_flags, diff, percent in coarsened_results
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load (image.pt, nap.json) from a selected folder and perform NAP coarsening."
    )
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--timeout', type=int, default=700, help='Timeout in seconds.')
    parser.add_argument('--selected_dir', type=str, required=True,
                        help='Folder containing image.pt and nap.json (optionally results.json).')
    parser.add_argument('--label', type=int, default=None,
                        help='Label to use (fallback if not found in results.json).')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Epsilon to use (fallback if not found in results.json).')
    parser.add_argument('--theta', type=float, default=0.2,
                        help='Stochastic coarsening probability per neuron.')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of stochastic iterations.')
    args = parser.parse_args()

    # Prepare experiment folder
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EXPERIMENTS_DIR, f"{script_name}+{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save params.json
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Capture stdout/stderr into log.txt
    log_buffer = StringIO()
    log_path = os.path.join(run_dir, "log.txt")

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            print(f"[INFO] Project root: {PROJECT_ROOT}")
            print(f"[INFO] Experiments dir: {run_dir}")
            print(f"[INFO] Selected dir: {args.selected_dir}")

            # Device
            device = "cuda" if (args.gpu and torch is not None and torch.cuda.is_available()) else "cpu"
            print(f"[INFO] Using device: {device}")

            # Init verifier + model
            verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
            try:
                model = load_model(args.model, device=device)
            except TypeError:
                model = load_model(args.model)
            _ = model  # kept for potential side effects/compat

            # Load (image_t, nap, label, epsilon)
            image_t, nap, label, epsilon = load_selected_run(
                args.selected_dir, device=device, fallback_label=args.label, fallback_epsilon=args.epsilon
            )
            print(f"[INFO] Using label={label}, epsilon={epsilon}")
            print(f"[INFO] tensor shape={tuple(image_t.shape)}, dtype={image_t.dtype}, device={image_t.device}")

            # Coarsen
            results = process_loaded_example(
                image_t=image_t,
                nap=nap,
                label=label,
                epsilon=epsilon,
                verifier=verifier,
                args=args,
            )

            # Save results.json
            results_path = os.path.join(run_dir, "results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"[OK] Results saved to {results_path}")

    except Exception as e:
        print(f"[ERROR] {e.__class__.__name__}: {e}")
        # Also write minimal results when failing
        results_path = os.path.join(run_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"error": f"{e.__class__.__name__}: {e}"}, f, indent=2, ensure_ascii=False)
    finally:
        # Flush log buffer to file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_buffer.getvalue())
        print(f"(logs in {log_path})")


if __name__ == "__main__":
    main()
