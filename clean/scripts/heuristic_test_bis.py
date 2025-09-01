import argparse
import sys
import time
from clean.nap_extraction.nap_utils import summarize_nap
from clean.verifier.verifier_base import Verifier
from clean.scripts.useful_func_bis import get_nap_specification_exclusive
from clean.nap_extraction.nap_utils import diff_naps, get_coarsening_percentage
from clean.nap_extraction.extract_nap_changed  import load_model
from contextlib import redirect_stdout
import os
import json
import torch


DEFAULT_HEURISTICS = ["simple", "random"]


# ---utils 

class Tee:
    """Écrit stdout et un fichier"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()

# end ---utils 
def _count_timeouts(timeout_flags):
    """
    Count number of True flags in possibly nested containers.
    - Handles: None, dict (values), list/tuple/set (possibly nested), scalars.
    - Typical case here: list of lists with booleans at [i][j].
    """
    if timeout_flags is None:
        return 0


    
    if isinstance(timeout_flags, (list, tuple, set)):
        total = 0
        for v in timeout_flags:
            if isinstance(v, (list, tuple, set, dict)):
                total += _count_timeouts(v)
            else:
                total += int(bool(v))
        return total

    
    return int(bool(timeout_flags))


def print_detailed(results):
    """
    Print a summary of heuristic results.

    results: list of dicts with keys:
      'Heuristic', 'coarsened'
    """
    if not results:
        print("[No results to display]")
        return

    print("\n===  Detailed Heuristics Summary ===")
    for row in results:
        print(f"\n\n")
        summarize_nap(row['coarsened'],row['Heuristic'])

def print_heuristics_table(results):
    """
    Print a table from heuristic results.

    results: list of dicts with keys:
      'Heuristic', 'Diff', 'Remaining%', 'Timeouts', 'Time(s)'
    """
    if not results:
        print("[No results to display]")
        return

    headers = ["Heuristic", "Diff", "Remaining%", "Timeouts", "Time(s)"]

    # compute max width per column
    col_widths = {}
    for h in headers:
        max_val_len = max(len(str(row[h])) for row in results)
        col_widths[h] = max(len(h), max_val_len)

    header_line = " │ ".join(f"{h:<{col_widths[h]}}" for h in headers)
    separator = "─" * len(header_line)

    print("\n=== Heuristics Results Table ===")
    print(header_line)
    print(separator)
    for row in results:
        print(" │ ".join(
            f"{str(row[h]):<{col_widths[h]}}" if h == "Heuristic"
            else f"{str(row[h]):>{col_widths[h]}}"
            for h in headers
        ))


def print_comparative_table(rows):
    """
    rows: list of dicts with keys:
      'Heuristic', 'Diff', 'Remaining%', 'Timeouts', 'Time(s)'
    """
    if not rows:
        print("\n=== Comparative Summary ===")
        print("[No results to display]")
        return

    h_w = max(len("Heuristic"), *(len(str(r["Heuristic"])) for r in rows))
    d_w = max(len("Diff"), *(len(str(r["Diff"])) for r in rows))
    p_w = len("Remaining %")
    t_w = len("Timeouts")
    s_w = len("Time (s)")

    header = (
        f"{'Heuristic':<{h_w}} │ "
        f"{'Diff':>{d_w}} │ "
        f"{'Remaining %':>{p_w}} │ "
        f"{'Timeouts':>{t_w}} │ "
        f"{'Time (s)':>{s_w}}"
    )
    sep = "─" * len(header)

    print("\n=== Comparative Summary ===")
    print(header)
    print(sep)
    for r in rows:
        # Ensure numeric formatting where applicable
        remaining_pct = r.get("Remaining%", float("nan"))
        time_s = r.get("Time(s)", float("nan"))
        print(
            f"{str(r['Heuristic']):<{h_w}} │ "
            f"{str(r['Diff']):>{d_w}} │ "
            f"{remaining_pct:>{p_w}.2f} │ "
            f"{str(r['Timeouts']):>{t_w}} │ "
            f"{time_s:>{s_w}.2f}"
        )




def build_parser():
    p = argparse.ArgumentParser(description="Run heuristic tests on NAP coarsening.")
    p.add_argument('--model', type=str, required=True, help='Path to the ONNX model.')
    p.add_argument('--heuristics', type=str, nargs='+', default=DEFAULT_HEURISTICS,
                   help='List of heuristics to test (default: simple random).')
    # Name of the experiment folder
    p.add_argument('--exp-name', type=str, default=None,
                   help='Name of the experiment (default: timestamp).')
    # Arguments used by Verifier
    p.add_argument('--json', type=str, default=None, help='Optional JSON spec/config for the verifier.')
    p.add_argument('--gpu', type=int, default=-1, help='GPU id to use, -1 for CPU.')
    p.add_argument('--timeout', type=int, default=120, help='Timeout (seconds) for verification calls.')

    # NAP spec arguments
    p.add_argument('--device', type=str, default='cpu', help='cpu | cuda')
    p.add_argument('--label', type=int, default=1, help='Target label for the NAP specification.')
    p.add_argument('--delta', type=float, default=0.001, help='Delta for NAP spec construction.')
    # build_parser()
    p.add_argument('--summary-path', type=str, default='heuristics_summary.txt',
               help='Chemin du fichier de résumé.')
  
    return p




def main():
    parser = build_parser()
    args = parser.parse_args()

    model_path = args.model
    # deduplication 
    heuristics = list(dict.fromkeys(args.heuristics))

    # Create verifier 
    try:
        verifier = Verifier(model_path, args.json, args.gpu, args.timeout)
    except TypeError:
        verifier = Verifier(model=model_path, json=args.json, gpu=args.gpu, timeout=args.timeout)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Verifier: {e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    try:
        model_loaded = load_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}", file=sys.stderr)
        sys.exit(1)

    label = args.label
    print(f"[INFO] Using model: {model_path} for label {label}")

    # Build exclusive NAP spec
    try:
        image, max_epsilon, nap = get_nap_specification_exclusive(
            label=label,
            verifier=verifier,
            model=model_loaded,
            args=args,
            device=args.device,
            delta=args.delta,
            return_all=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to build NAP specification: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":

    main()
