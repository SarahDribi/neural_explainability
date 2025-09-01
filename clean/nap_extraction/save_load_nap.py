"""
Here I should write the functions that save /load a nap from a certain location 
"""
import json

def save_nap_to_json(nap, filename):
    with open(filename, 'w') as f:
        json.dump(nap, f)
    print(f"[INFO] Coarsened NAP saved to {filename}")


def load_nap_from_json(filename):
    with open(filename, 'r') as f:
        nap = json.load(f)
    print(f"[INFO] Coarsened NAP loaded from {filename}")
    return nap
