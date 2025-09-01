
from clean.nap_extraction.extract_nap import nap_extraction_from_onnx
from clean.dataset.utils import get_label_inputs

def max_epsilon_robustness(x, nap, label, verifier, low=0.01, high=0.4, tol=0.001, max_iter=60):
    
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

def max_epsilon_nap_robustness(x, label, verifier, low=0.01, high=0.4, tol=0.001, max_iter=60):
    
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
 











 # finding good inputs that have certain robustness properties 


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

        
        max_eps_nap = max_epsilon_robustness(
            image_input, nap, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )

        
        max_eps_base = max_epsilon_nap_robustness(
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

   
