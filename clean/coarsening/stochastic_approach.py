

"""
A faster approach :
Stochastic , to avoid a one by one selection at a time


"""
import numpy as np
def sample_nap(nap, probability):
    copy = [layer.copy() for layer in nap]  # deep copy
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if np.random.rand() < probability:
                copy[i][j] = -1  # abstract neuron with probability theta
    return copy




def stochasticShorten(nap, input, label, epsilon, verifier, theta, max_iterations=15):
    initial_nap = [layer.copy() for layer in nap]  
    initial_success = 0
      
    
    verification_result = verifier.is_verified_nap(nap, input, label, epsilon)
    if not verification_result:
        print("[INFO] NAP is not robust initially.")
        return None  # initial NAP is not verified, no need to shorten

    for it in range(max_iterations):
        sampled_nap = sample_nap(nap, probability=theta)
        verification_result,_ = verifier.is_verified_nap_small_timeout(nap, input, label, epsilon)
        if verification_result:
            print(f"[INFO] Coarsened NAP at iteration {it} was verified as robust.")
            nap = sampled_nap 
            initial_success+=1

    return nap,initial_success
