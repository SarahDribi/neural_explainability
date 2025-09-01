import argparse
import time

from plnn.proxlp_solver.propagation import Propagation
from plnn.explp_solver.solver import ExpLP
from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
from plnn.branch_and_bound.branching_scores import BranchingChoice
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.custom_torch_modules import Flatten
from plnn.model import simplify_network

import torch, copy

import json

"""

    Contains an introduction to the functionalities of the codebase: how to load an ONNX network, how to compute 
    intermediate and output bounds, how to run a basic branch-and-bound instance without resorting to .json 
    configuration files.

"""
#returns a list of the naps that i need 
# 1 means a neuron is activated 
# 0 means a neuron isn t 
# -1 means we arent sure about its state
def get_label_nap(mined_Naps,label=0):
    # I should replace this with the corresponding 
    #computed NAP
    #return [[1,0,-1],[1,1,1]]
    print(f"the mined label is {mined_Naps[label]} ")
    return mined_Naps[label]









def generate_tiny_random_cnn():
    # Generate a very small CNN with random weight for testing purposes.
    # Input dimensions.
    in_chan = 3
    in_row = 2
    in_col = 2
    #Generate input domain.
    input_domain = torch.zeros((in_chan, in_row, in_col, 2))
    in_lower = (20 - -10) * torch.rand((in_chan, in_row, in_col)) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand((in_chan, in_row, in_col)) + (in_lower + 1)
    input_domain[:, :, :, 0] = in_lower
    input_domain[:, :, :, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_chan_c1 = 7
    ker_size = 2
    conv1 = torch.nn.Conv2d(in_chan, out_chan_c1, ker_size, stride=2, padding=1)
    conv1.weight = torch.nn.Parameter(torch.randn((out_chan_c1, in_chan, ker_size, ker_size)), requires_grad=False)
    conv1.bias = torch.nn.Parameter(torch.randn(out_chan_c1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    ker_size = 2
    out_chan_c2 = 5
    conv2 = torch.nn.Conv2d(out_chan_c1, out_chan_c2, ker_size, stride=5, padding=0)
    conv2.weight = torch.nn.Parameter(torch.randn((out_chan_c2, out_chan_c1, ker_size, ker_size)), requires_grad=False)
    conv2.bias = torch.nn.Parameter(torch.randn(out_chan_c2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_chan_c2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_chan_c2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)
    layers = [conv1, relu1, conv2, relu2, Flatten(), final]

    return layers, input_domain


def generate_tiny_random_linear(precision):
    # Generate a very small fully connected network with random weight for testing purposes.
    # Input dimensions.
    input_size = 2
    #Generate input domain.
    input_domain = torch.zeros((input_size, 2))
    in_lower = (20 - -10) * torch.rand(input_size) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand(input_size) + (in_lower + 1)
    input_domain[:, 0] = in_lower
    input_domain[:, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_size1 = 3
    lin1 = torch.nn.Linear(input_size, out_size1)
    lin1.weight = torch.nn.Parameter(torch.randn((out_size1, input_size)), requires_grad=False)
    lin1.bias = torch.nn.Parameter(torch.randn(out_size1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    out_size2 = 3
    lin2 = torch.nn.Linear(out_size1, out_size2)
    lin2.weight = torch.nn.Parameter(torch.randn((out_size2, out_size1)), requires_grad=False)
    lin2.bias = torch.nn.Parameter(torch.randn(out_size2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_size2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_size2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)

    input_domain = (input_domain).type(precision)
    lin1.weight = torch.nn.Parameter(lin1.weight.type(precision))
    lin1.bias = torch.nn.Parameter(lin1.bias.type(precision))
    lin2.weight = torch.nn.Parameter(lin2.weight.type(precision))
    lin2.bias = torch.nn.Parameter(lin2.bias.type(precision))
    final.weight = torch.nn.Parameter(final.weight.type(precision))
    final.bias = torch.nn.Parameter(final.bias.type(precision))

    layers = [lin1, relu1, lin2, relu2, final]

    return layers, input_domain


def parse_input(precision=torch.float):
    # Parse the input specifications: return network, domain, args

    torch.manual_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_filename', type=str, help='onnx file to load.')
    parser.add_argument('--random_net', type=str, choices=["cnn", "linear"], help='whether to use a random network')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--init_step', type=float, default=1e0, help="step size for optimization based algos")

    args = parser.parse_args()

    if args.random_net and args.network_filename:
        raise IOError("Test either on a random network or on a .rlv, not both.")

    if args.network_filename:
        # Test on loaded ONNX net.
        # For instance: ./models/onnx/cifar_base_kw.onnx
        assert args.network_filename.endswith('.onnx')

        model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)

        if not model_correctness:
            return None, False

        # Assert that the model specification is currently supported.
        supported = vnnlib_utils.is_supported_model(model)
        assert supported

        eps = 0.1
        # Define input bounds as being a l_inf ball of eps around a randomly sampled point in [0, 1]
        input_point = torch.rand(in_shape)
        input_bounds = torch.stack([(input_point - eps).clamp(0, 1), (input_point + eps).clamp(0, 1)], dim=-1)

        # ReLUify any maxpool...
        with torch.no_grad():
            layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype=dtype)

    else:
        if args.random_net == "cnn":
            # Test on small generated CNN.
            layers, input_bounds = generate_tiny_random_cnn()
        else:
            # Test on small generated fully connected network.
            layers, input_bounds = generate_tiny_random_linear(precision)

    return layers, input_bounds, args
def update_bounds_using_nap(upper_bounds,lower_bounds,label,mined_naps):
    nap=get_label_nap(mined_naps,label)
    print(f"the Nap activation pattern is {nap}")
    for i in range(1,len(upper_bounds)-1):
        for j in range(len(upper_bounds[i][0])):
             # now we are going throught all the neurons 
             if(nap[i-1][j]==1):# if that neuron is active 
                #set its lower bound to 0
                lower_bounds[i][0][j]=0
             elif(nap[i-1][j]==0):
                 # means that the neuron  is inactive 
                 #set its relu upper bound to 0
                 upper_bounds[i][0][j]=0
             




def compute_bounds_for_Nap(label,mined_naps):

    precision = torch.float

    # A network is expressed as list of torch and custom layers (custom layers defined in tools/custom_torch_modules.py)
    layers, domain, args = parse_input(precision=precision)

    # make the input domain a batch of domains -- in this case of size 1
    batch_domain = domain.unsqueeze(0)

    gpu = args.gpu
    if gpu:
        # the copy is necessary as .cuda() acts in place for nn.Parameter
        exp_layers = [copy.deepcopy(lay).cuda() for lay in layers]
        exp_domain = batch_domain.cuda()
    else:
        exp_layers = layers
        exp_domain = batch_domain
    
 
    # Given the network, the l_inf input domain specified as a stack (over the last dimension) of lower and upper
    # bounds over the input, the various bounding classes defined in ./plnn/ create an internal representation of the
    # network with the classes defined in plnn.proxlp_solver.utils

    # The internal representation is internally built in plnn.dual_bounding, which defines the DualBounding that all
    # bounding methods inherit from. In particular, the conversion occurs within define_linear_approximation or
    # within build_model_using_bounds

    # Compute intermediate bounds using best bounds between CROWN and KW
    intermediate_net = Propagation(exp_layers, type="best_prop", params={"best_among": ["KW", "crown"]}, max_batch=2000)
    # intermediate_net = Propagation(exp_layers, type="naive", max_batch=2000)  # uses IBP bounds -- much looser
    with torch.no_grad():
        intermediate_net.define_linear_approximation(exp_domain)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds
    first_lower_bounds=intermediate_lbs
    first_upper_bounds=intermediate_ubs
    print (f"the initial  intermediate upper bounds are ",intermediate_ubs)
    print(f"\n\n")
    print(f"the intial intermediate lower bounds  are  ",intermediate_lbs)
    print(f"\n\n")
    # here i am going to edit them like this :
    "if I know the neuron of layer i is active:   "
    #transform its corresponding lower bound to be 0

    "if I know the neuron of layer i is inactive "
    #transform its corresponding upper_bound to be 0
    
    update_bounds_using_nap(intermediate_ubs,intermediate_lbs,label,mined_naps)
    print (f"the updated intermediate upper bounds are ",intermediate_ubs)
    print(f"\n\n")
    print(f"the updated lower bounds  are  ",intermediate_lbs)
    print(f"\n\n")
    check_res=sanity_check(first_lower_bounds,intermediate_lbs,first_upper_bounds,intermediate_ubs)
    print(f"running a little sanity check for label {label }'s bounds :\n The final bounds should become tighter {check_res}")
    return intermediate_lbs,intermediate_ubs

    
# should check that bounds became tighter 
# because we imposed more constraints

def is_tigher_final_upper_bound(first_upper_bounds,updated_upper_bounds):
    #if a value from updated is higher return false 
    last_bound=len(first_upper_bounds)-1
        #parcours des la bound
    for j in range (len(first_upper_bounds[last_bound][0])):
            print(f"the {j}th element is {first_upper_bounds[last_bound][0][j]}")
            if first_upper_bounds[last_bound][0][j]<updated_upper_bounds[last_bound][0][j]:
                
                return False
                

    return True

def is_tigher_final_lower_bound(first_lower_bounds,updated_lower_bounds):
       #if a value from updated lower_bound is lower  return false 
    last_bound=len(first_lower_bounds)-1
        #parcours des inner bounds
    for j in range (len(first_lower_bounds[last_bound][0])):
            print(f"the {j}th element is {first_lower_bounds[last_bound][0][j]}")
            if first_lower_bounds[last_bound][0][j]>updated_lower_bounds[last_bound][0][j]:
                return False
                

    return True



def sanity_check(first_lower_bounds,updated_lower_bounds,first_upper_bounds,updated_upper_bounds):
    #check that lower bounds are tigher
    if not is_tigher_final_lower_bound(first_lower_bounds,updated_lower_bounds):
        return False
    if not is_tigher_final_upper_bound(first_upper_bounds,updated_upper_bounds):
        return False



    #check that upper bounds are tigher
    return True 







"""
sample = label_to_samples[3][4500]  # entry for image of label 3
nap_template = labelNaps[3]         # Nap for label 3
is_match = follows_nap_array(sample, model, nap_template)

print("Sample follows label 3 NAP:" if is_match else "Sample does NOT follow label 3 NAP.")



"""



# Load NAPs from file
nap_file = "mined_naps_small_net.json"
with open(nap_file, "r") as f:
    labelNaps = json.load(f)

if __name__ == '__main__':

    
    
    compute_bounds_for_Nap(1,labelNaps)


    #print(labelNaps)
    print(labelNaps[0])
