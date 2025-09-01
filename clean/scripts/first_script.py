
import argparse
import json
import os
import sys
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Resolving relative paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Make project_root importable regardless of current working directory
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Imports from your project (now that sys.path is set) ---
from clean.coarsening.nap_coarsen import shorten_nap_around_input, shorten_nap_around_input_complex_heuristic
from clean.coarsening.stochastic_approach import stochasticShorten
from clean.nap_extraction.nap_utils import diff_naps, get_coarsening_percentage
from clean.nap_extraction.extract_nap import load_model
from clean.heuristics.simple_order_heur import simple_order_neurons
from clean.heuristics.random_order_heur import get_shuffled_neurons
from clean.scripts.useful_func import get_nap_specification_exclusive
from clean.verifier.verifier_base import Verifier


def process_nap_exclusive_example(label, verifier, model, args, device):
    image, epsilon, nap = get_nap_specification_exclusive(
        label=label,
        verifier=verifier,
        model=model,
        args=args,
        device=device,
        delta=0.003,
        return_all=False
    )
    print(f"[INFO] Found NAP-exclusive robust input for label {label} at epsilon={epsilon:.4f}")

    # Example deterministic heuristics (empty by default—uncomment to use)
    simple_heuristics = [simple_order_neurons]  # [simple_order_neurons, get_shuffled_neurons]
    simple_heur_names = ["simple"]  # ["simple", "random"]

    coarsened_results = []
    for i, heuristic in enumerate(simple_heuristics):
        coarsened,timeout_flags= shorten_nap_around_input(nap, image, label, epsilon, heuristic, verifier)
        diff = diff_naps(nap, coarsened)
        percent = get_coarsening_percentage(nap, coarsened)
        coarsened_results.append((simple_heur_names[i], coarsened,timeout_flags, diff, percent))

    # Stochastic trials (one run by default; use args.theta / args.iterations)
    coarsened, successful_iterations = stochasticShorten(
        nap, image, label, epsilon, verifier, args.theta, args.iterations
    )
    diff_val = diff_naps(nap, coarsened)
    percent_val = get_coarsening_percentage(nap, coarsened)
    print(f"[Stochastic] Coarsened neurons: {diff_val} | Remaining %: {percent_val:.2f} "
          f"| successful_iterations={successful_iterations}")

    print(f"[INFO] Coarsened NAP: {coarsened}")
    print(f"[INFO] Original NAP:  {nap}")

    return {
        "label": int(label),
        "epsilon": float(epsilon),
        "stochastic_diff": int(diff_val),
        "stochastic_remaining_percent": float(percent_val),
        "stochastic_successful_iterations": int(successful_iterations),
        "coarsened_nap": coarsened,
        "original_nap": nap,
        "coarsened_results": [
            {
                "heuristic": name,
                "coarsened_nap": coarsened_nap,
                "timeout_flags": timeout_flags,
                "diff": diff,
                "remaining_percent": percent
            }
            for name, coarsened_nap, timeout_flags, diff, percent in coarsened_results
        ]

    }


def main():
    parser = argparse.ArgumentParser(description="Test verifier on NAP-exclusive robustness and log results.")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--json', type=str, required=False, help='Optional BaB config JSON file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for verification.')
    parser.add_argument('--label', type=int, required=True, help='Label to verify (NAP of label l_i).')
    parser.add_argument('--timeout', type=int, default=700, help='Timeout in seconds.')
    parser.add_argument('--theta', type=float, default=0.2, help='Stochastic coarsening probability per neuron.')
    parser.add_argument('--iterations', type=int, default=100, help='Number of stochastic iterations.')
    args = parser.parse_args()

    # --- Prepare experiment folder ---
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]  # e.g., "verify_nap_exclusive"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EXPERIMENTS_DIR, f"{script_name}+{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save params.json
    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Capture stdout/stderr into log.txt
    log_buffer = StringIO()
    log_path = os.path.join(run_dir, "log.txt")

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            print(f"[INFO] Project root: {PROJECT_ROOT}")
            print(f"[INFO] Experiments dir: {run_dir}")

            verifier = Verifier(args.model, args.json, args.gpu, args.timeout)
            model = load_model(args.model)

            results = process_nap_exclusive_example(
                label=args.label,
                verifier=verifier,
                model=model,
                args=args,
                device="cpu",
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
        # Also echo where the logs are stored to console
        print(f"(logs in {log_path})")


if __name__ == "__main__":
    main()

    
    
    
    
  
    
    
    
    
    
    
    
    
    
   
    
    
    
  
    