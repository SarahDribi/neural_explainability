"""
d logit[label]/dpre-activation_neuron  
on coupe avant le ReLU ciblé, on traite la préactivation z comme variable,
on propage le suffixe, puis on rétro-propage jusqu'à z
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from tools.bab_tools import vnnlib_utils


# ---------- Chargement modèle ONNX -> nn.Module ----------
def load_model(onnx_path: str) -> nn.Module:
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    print(f"[INFO] Loading ONNX model: {onnx_path}")
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    if not model_correct or model is None:
        raise RuntimeError("ONNX to torch model conversion failed")
    return model.eval()


# ---------- Utilitaires : ordre des modules + forward découpé ----------
def _modules_in_order(model: nn.Module) -> List[nn.Module]:
    if isinstance(model, nn.Sequential):
        return list(model)
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        return list(model.net)
    return list(model.children())

def _forward_cut_at_relu(
    nn_model: nn.Module, x: torch.Tensor, relu_layer_idx: int
) -> Tuple[torch.Tensor, List[nn.Module], int]:
    modules = _modules_in_order(nn_model)
    if not modules:
        raise RuntimeError("Modèle non séquentiel ou vide: impossible d'itérer les couches.")

    state = x
    relu_count = 0
    cut_index = None

    for idx, layer in enumerate(modules):
        if isinstance(layer, nn.ReLU):
            if relu_count == relu_layer_idx:
                cut_index = idx
                break
            state = layer(state)
            relu_count += 1
        else:
            state = layer(state)

    if cut_index is None:
        raise IndexError(f"relu_layer_idx={relu_layer_idx} introuvable (ReLU rencontrés: {relu_count}).")

    z = state
    suffix_layers = modules[cut_index:]  # inclut le ReLU ciblé
    return z, suffix_layers, cut_index

def _forward_suffix(suffix_layers: List[nn.Module], z: torch.Tensor) -> torch.Tensor:
    out = z
    for l in suffix_layers:
        out = l(out)
    return out

def _select_logit(out: torch.Tensor, label: int) -> torch.Tensor:
    if out.dim() == 2 and out.shape[0] == 1:
        return out[0, label]
    elif out.dim() == 1:
        return out[label]
    else:
        return out.reshape(1, -1)[0, label]


# ---------- Dérivée locale pour 1 neurone ----------
def get_preactivation_derivative_for_logit(
    neuron: Tuple[int, int],
    nn_model: nn.Module,
    x: torch.Tensor,
    label: int,
) -> float:
    """
    | d logit[label] / d z_i | au point x,
    où z_i est l'élément 'neuron_idx' de la préactivation avant le ReLU 'relu_layer_idx'.
    """
    if x.shape[0] != 1:
        raise ValueError("x doit avoir un batch de taille 1.")
    relu_layer_idx, neuron_idx = neuron
    if not isinstance(relu_layer_idx, int) or not isinstance(neuron_idx, int):
        raise TypeError("neuron doit être un tuple (int, int).")

    nn_model.eval()
    with torch.enable_grad():
        z, suffix_layers, _ = _forward_cut_at_relu(nn_model, x, relu_layer_idx)
        z = z.detach().clone().requires_grad_(True)
        out = _forward_suffix(suffix_layers, z)
        if label < 0 or label >= out.reshape(1, -1).shape[1]:
            raise IndexError(f"label={label} hors bornes pour logits de taille {out.reshape(1,-1).shape[1]}")
        y = _select_logit(out, label)
        grad_z, = torch.autograd.grad(y, z, retain_graph=False, create_graph=False, allow_unused=False)

    gflat = grad_z.view(-1)
    if not (0 <= neuron_idx < gflat.numel()):
        raise IndexError(f"neuron_idx={neuron_idx} hors bornes (taille préactivation: {gflat.numel()}).")
    return float(gflat[neuron_idx].abs().item())


# ---------- Lister (relu_idx, neuron_idx) avec un x pour connaître les shapes ----------
def get_neurons_from_model(
    nn_model: nn.Module,
    x: torch.Tensor,
    active_only: bool = False,
) -> List[Tuple[int, int]]:
    if x.shape[0] != 1:
        raise ValueError("x must have batch size 1.")
    nn_model.eval()

    modules = _modules_in_order(nn_model)
    if not modules:
        raise RuntimeError("Cannot iterate layers; is the model sequential?")

    neurons: List[Tuple[int, int]] = []
    state = x
    relu_idx = -1

    with torch.no_grad():
        for layer in modules:
            if isinstance(layer, nn.ReLU):
                relu_idx += 1
                z = state  # préactivation avant ce ReLU
                z0 = z[0] if z.dim() > 1 else z
                if active_only:
                    idxs = z0.flatten().gt(0).nonzero(as_tuple=False).view(-1).tolist()
                else:
                    idxs = list(range(z0.numel()))
                neurons.extend((relu_idx, j) for j in idxs)
                state = layer(state)  # on applique le ReLU pour continuer
            else:
                state = layer(state)

    return neurons


# ---------- Ordonner par impact décroissant ----------
def higher_impact_order_in_memory(
    neurons: List[Tuple[int, int]],
    onnx_path: str,
    x_input: torch.Tensor,
    x_label: int,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    importance_score: Dict[Tuple[int, int], float] = {}
    
    input=x_input.unsqueeze(0)
    try:
        nn_model= load_model(onnx_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {onnx_path}: {e}", file=sys.stderr)
        sys.exit(1)

    for neuron in neurons:
        s = get_preactivation_derivative_for_logit(neuron, nn_model, input, x_label)
        importance_score[neuron] = s
    sorted_neurons = sorted(neurons, key=lambda n: importance_score[n], reverse=True)
    return sorted_neurons, importance_score


# ---------- Donnée MNIST (batch=1,1,28,28) ----------
def get_label_input(label: int) -> torch.Tensor:
    """
    Renvoie un tenseur batché [1, 1, 28, 28] pour la classe 'label'.
    (mon doute: ton modèle exige-t-il du 4D ou du 2D ? ici on choisit 4D car beaucoup de modèles ont Flatten interne)
    """
    transform = Compose([ToTensor()])  # -> [1,28,28] float32 in [0,1]
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    for x, y in dataset:
        if int(y) == int(label):
            return x.unsqueeze(0)  # [1,1,28,28]
    raise RuntimeError(f"Aucune image trouvée pour le label {label}")


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank neurons by pre-activation impact on a target logit.")
    p.add_argument("--model", required=True, help="Path to ONNX model")
    p.add_argument("--label", type=int, default=9, help="Target class index (default: 9)")
    p.add_argument("--active-only", action="store_true", help="Keep only neurons with z>0 at x")
    p.add_argument("--topk", type=int, default=70, help="Print top-k neurons")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    nn_model= load_model(args.model)
    # 2) Préparer une image MNIST du label choisi (format batch 4D)
    x = get_label_input(args.label)  # [1,1,28,28]
    # (si ton modèle réclame un vecteur [1,784], remplace par: x = x.view(1, 784))

    # 3) Lister les neurones
    neurons = get_neurons_from_model(nn_model, x, active_only=args.active_only)
    if not neurons:
        print("[WARN] Aucun neurone ReLU détecté.", file=sys.stderr)
        sys.exit(0)

    # 4) Ordonner par impact décroissant
    ordered, scores = higher_impact_order_in_memory(neurons, args.model, x, args.label)

    # 5) Afficher top-k
    topk = min(args.topk, len(ordered))
    print(f"\nTop-{topk} neurones par importance (|∂logit[{args.label}] / ∂z|) :")
    for n in ordered[:topk]:
        print(f"  neuron={n}   score={scores[n]:.6g}")


if __name__ == "__main__":
    main()
