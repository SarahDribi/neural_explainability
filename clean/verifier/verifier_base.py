from clean.verifier.nap_constrained_verif import  verify_nap_property_around_input
from clean.verifier.region_robustness_verif import verify_robustness_around_input

# The Verifier class encapsulates two types of verification:

     # is_verified_nap: checks whether an input x' remains correctly classified 
# within an epsilon-ball around  x and following a fixed (NAP) x (Nap).
# Similar to Nap augmented specification 
# Robustness Meaning no adversarial example following x 's Nap sould exist
#################################################################

    # is_verified_region: checks standard robustness =>whether there exists 
# an adversarial example within an epsilon-ball around the input.
# make it more clear

class Verifier:
    def __init__(self, model_path, json_config, use_gpu, timeout):
        self.model_path = model_path
        self.json_config = json_config
        self.timeout = timeout
        self.use_gpu = use_gpu
        

    def is_verified_nap(self, nap,input, label, epsilon):
        result,timed_out = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            
            nap=nap,
            label=label,
            use_gpu=False,
            timeout=self.timeout
            
    
        )
        return result # use with a big timeout
    def is_verified_nap_small_timeout(self, nap, input, label, epsilon):
        result, timed_out = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            nap=nap,
            label=label,
            use_gpu=False,
            timeout=12 # small timeout for fast verification
        )
        return result, timed_out  # return both result and timeout status

    def is_verified_region(self,input,label,epsilon):
        result=verify_robustness_around_input(model_path=self.model_path,input=input,epsilon=epsilon,json_config=self.json_config, label=label, use_gpu=False,timeout=self.timeout)
        return  result


# Define a ready to use verifier 
# A verifier for the Small model


