 
    #In order to  coarsen effectively the Nap in an epsilon Ball around input


   # We need to find the largest robust epsilon ball around an input image such that 

   # for the region around the  it is robust with the nap specification 
   #   but it s not strong without the Nap specification 
from utils import (
    load_model,
    TrackedActivationsModel,
    get_label_input,
    get_label_inputs,
    nap_extraction,
    nap_extraction_from_onnx,
    display_nap_array,
    diff_naps,
    get_coarsening_percentage


)
def find_max_epsilon_around_input(x, nap, label, verifier, low=0.01, high=0.4, tol=0.001, max_iter=60):
    
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.is_verified_nap(nap,x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.is_verified_nap(nap,x, label, low):
        print("[WARN Nap ]Sorry to tell that it was not found to be robust even here {low}")
        return low*(-1)
    return low

def find_max_epsilon_around_input_without_nap(x, label, verifier, low=0.01, high=0.4, tol=0.001, max_iter=60):
    
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.is_verified_region(x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.is_verified_region(x, label, low):
        print(f"[WARN]Sorry to tell that it was not found to be robust even here {low}")
        return low*(-1)
    return low
 











 # finding inputs with certain robustness properties 
 
# Fonction principale de recherche d'un input nap-epsilon-robuste
def find_input_robust_at_epsilon(
    eps: float,
    label: int,
    args,
    verifier,
    model,
    device: str = "cpu"
):
    """
    Recherche un input du label donné qui est correctement classé
    par le modèle et epsilon-robuste vis-à-vis de sa NAP.

    Returns:
        (index, image_input, nap) si trouvé, sinon (-1, None, None)
    """
    
    inputs = get_label_inputs(label)
    print(f"[INFO] Searching for robust input for label {label} and epsilon={eps}")

    for i in range(0, len(inputs)):
        image_input = inputs[i].to(device).unsqueeze(0)

        # Prédiction du modèle
        prediction = model(image_input).argmax(dim=1).item()
        if prediction != label:
            continue

        # Extraction NAP
        nap = nap_extraction_from_onnx(args.model, image_input.squeeze(0))

        # Vérification robustesse
        
        if verifier.is_verified_nap(nap, image_input, label, eps):
            print(f"[SUCCESS] Robust input found at index {i}")
            return i, image_input.squeeze(0), nap

    print("[FAILURE] No robust input found for given epsilon.")
    return -1, None, None


def find_input_robust_at_epsilon_without_nap(
    eps: float,
    label: int,
    args,
    verifier,
    model,
    device: str = "cpu"
):
    """
    Recherche un input du label donné qui est correctement classé
    par le modèle et epsilon-robuste vis-à-vis de sa NAP.

    Returns:
        (index, image_input, nap) si trouvé, sinon (-1, None, None)
    """
    
    inputs = get_label_inputs(label)
    print(f"[INFO] Searching for robust input for label {label} and epsilon={eps}")

    for i in range(0, len(inputs)):
        image_input = inputs[i].to(device).unsqueeze(0)

        # Prédiction du modèle
        prediction = model(image_input).argmax(dim=1).item()
        if prediction != label:
            continue

        # Extraction NAP
        nap = nap_extraction_from_onnx(args.model, image_input.squeeze(0))

        # Vérification robustesse
        if verifier.is_verified_nap(nap,image_input, label, eps):
            print(f"[SUCCESS] Robust input found at index {i}")
            return i, image_input.squeeze(0), nap

    print("[FAILURE] No robust input in the epsilon ball found for given epsilon.")
    return -1, None, None


def get_nap_specification_exclusive(
    label: int,
    verifier,
    model,
    args,
    device: str = "cpu",
    delta: float = 0.001,
    return_all: bool = False
):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    inputs = get_label_inputs(label)  
    print(f"[INFO] Searching for NAP-exclusive robust inputs for label {label}")

    results = []

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

    if return_all:
        return results
    else:
        print("No example found where NAP strictly improves robustness.")
        return None

   