import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from tools.bab_tools import vnnlib_utils
import random
import numpy as np

# Charger le modèle ONNX en torch
def load_model(onnx_path: str):
    model, _, _, _, model_correct = vnnlib_utils.onnx_to_pytorch(onnx_path)
    assert model_correct
    return model.eval()

# Classe avec suivi des activations ReLU
class TrackedActivationsModel(torch.nn.Module):
    def __init__(self, base, relu_layers=[2, 4]):
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

# Extraction NAP simple
def nap_extraction(model: TrackedActivationsModel, x: torch.Tensor):
    _ = model(x.unsqueeze(0))
    nap = []
    for name in sorted(model.activations.keys()):
        act = model.activations[name].flatten()
        nap.append([1 if v.item() > 0 else 0 for v in act])
    return nap

# Charger une image du label donné
def get_label_input(label, model):
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root="./data", train=True, transform=transform, download=True)

    for x, y in dataset:
        if y == label:
            x = x.view(-1)
            pred = model(x.unsqueeze(0)).argmax().item()
            if pred == label:
                return x
    raise RuntimeError(f"[FAIL] Aucun exemple correctement classé trouvé pour le label {label}")


# Comparaison entre deux NAP
def same_nap(nap1, nap2):
    return all(a == b for l1, l2 in zip(nap1, nap2) for a, b in zip(l1, l2))

# Expérience de stabilité de NAP
def test_nap_stability(model_path, label=1, epsilon=0.02, n_samples=50):
    print(f"[INFO] Loading model: {model_path}")
    base = load_model(model_path)
    model_tracked = TrackedActivationsModel(base)

    x = get_label_input(label, base)
    nap_ref = nap_extraction(model_tracked, x)
    pred_ref = base(x.unsqueeze(0)).argmax().item()

    count_same_nap = 0
    count_same_label = 0
    count_both = 0

    for i in range(n_samples):
        delta = (torch.rand_like(x) * 2 - 1) * epsilon
        x_pert = (x + delta).clamp(0, 1)

        nap_pert = nap_extraction(model_tracked, x_pert)
        pred_pert = base(x_pert.unsqueeze(0)).argmax().item()

        same_nap_flag = same_nap(nap_ref, nap_pert)
        same_label_flag = (pred_pert == pred_ref)

        if same_nap_flag:
            count_same_nap += 1
        if same_label_flag:
            count_same_label += 1
        if same_nap_flag and same_label_flag:
            count_both += 1

    print(f"\n[RESULTS for ε = {epsilon}, label = {label}]")
    print(f"→ {count_same_nap}/{n_samples} perturbations ont la même NAP")
    print(f"→ {count_same_label}/{n_samples} perturbations ont le même label")
    print(f"→ {count_both}/{n_samples} perturbations ont la même NAP ET le même label")

    if count_same_nap > 0:
        print(f"→ Parmi les {count_same_nap} NAP identiques, {count_both} ont gardé le label ({100*count_both/count_same_nap:.2f}%)")


if __name__ == "__main__":
    test_nap_stability("tools/mnist-10x2.onnx", label=1, epsilon=0.5, n_samples=6000)



# I should think about a way to use this as a test
# use as a predictor for the Naps volume inside the epsilon ball
# how to use it as a metric to compare the small and big network
# comment faire un petit comparatif des résultats entre the big and small net
# ordering strategies
