from utils_verif import  verify_nap_property_around_input
from utils_verif import verify_robustness_around_input

# The Verifier class encapsulates two types of verification:

     # is_verified_nap: checks whether an input x' remains correctly classified 
# within an epsilon-ball around  x and following a fixed (NAP) x (Nap).
# Similar to Nap augmented specification 
# Robustness Meaning no adversarial example following x 's Nap sould exist
#################################################################

    # is_verified_region: checks standard robustness =>whether there exists 
# an adversarial example within an epsilon-ball around the input.


class Verifier:
    def __init__(self, model_path, json_config, use_gpu, timeout):
        self.model_path = model_path
        self.json_config = json_config
        self.timeout = timeout
        self.use_gpu = use_gpu
        

    def is_verified_nap(self, nap,input, label, epsilon):
        result = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            
            nap=nap,
            label=label,
            use_gpu=False,
            timeout=self.timeout
            
    
        )
        return not result
    def is_verified_region(self,input,label,epsilon):
        result=verify_robustness_around_input(model_path=self.model_path,input=input,epsilon=epsilon,json_config=self.json_config, label=label, use_gpu=False,timeout=self.timeout)
        return not result


# Define a ready to use verifier 
# A verifier for the Small model

