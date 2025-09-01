import copy
import random
from nap_extraction_tris import Verifier, get_label_input, nap_extraction_from_onnx,stochasticShorten,diff_naps
import json
# === CONFIGURATION ===
MODEL_PATH = "tools/mnist-net_256x4.onnx"
LABEL = 2
JSON_CONFIG = None
EPSILON = -1 # means all the input space
TIMEOUT = 300

def save_nap_to_json(nap, filename):
    with open(filename, 'w') as f:
        json.dump(nap, f)
    print(f"[INFO] Coarsened NAP saved to {filename}")

def load_test_objects():
    verifier = Verifier(MODEL_PATH, JSON_CONFIG, use_gpu=False, timeout=TIMEOUT)
    input_image = get_label_input(LABEL)
    initial_nap = nap_extraction_from_onnx(MODEL_PATH, input_image)
    return verifier, input_image, initial_nap
def load_test_objects(label):
    verifier = Verifier(MODEL_PATH, JSON_CONFIG, use_gpu=False, timeout=TIMEOUT)
    input_image = get_label_input(label)
    initial_nap = nap_extraction_from_onnx(MODEL_PATH, input_image)
    return verifier, input_image, initial_nap
def test_full_abstraction_fails():
    print("\nRunning test_full_abstraction_fails...")
    verifier, input_image, initial_nap = load_test_objects()

    fully_abstracted_nap = [[-1 for _ in layer] for layer in initial_nap]

    is_verified = verifier.is_verified_nap(fully_abstracted_nap, input_image, LABEL, EPSILON)

    assert not is_verified, "Fully abstracted NAP should not be robust."
    print("[PASS] Fully abstracted NAP correctly identified as non-robust.")

def test_initial_nap_is_robust():
    print("\nRunning test_initial_nap_is_robust...")
    verifier, input_image, initial_nap = load_test_objects()

    is_verified = verifier.is_verified_nap(initial_nap, input_image, LABEL, EPSILON)

    assert is_verified, "Initial NAP should be robust."
    print("[PASS] Initial NAP correctly verified as robust.")


def test_verification_with_wrong_label_fails():
    print("\nRunning test_verification_with_wrong_label_fails...")
    verifier, input_image, initial_nap = load_test_objects()

    wrong_label = (LABEL + 1) % 10  # another different label

    is_verified = verifier.is_verified_nap(initial_nap, input_image, wrong_label, EPSILON)

    assert not is_verified, "Verification with wrong label should not pass."
    print("[PASS] Verification with wrong label correctly failed.")

def test_small_random_abstraction_should_remain_robust():
    print("\nRunning test_small_random_abstraction_should_remain_robust...")
    verifier, input_image, initial_nap = load_test_objects()

    test_nap = copy.deepcopy(initial_nap)
    all_neurons = [(i, j) for i in range(len(test_nap)) for j in range(len(test_nap[i]))]
    random.shuffle(all_neurons)

    num_to_abstract = max(1, int(0.01 * len(all_neurons)))  # Abstracting only ~1%
    for idx in range(num_to_abstract):
        i, j = all_neurons[idx]
        test_nap[i][j] = -1

    is_verified = verifier.is_verified_nap(test_nap, input_image, LABEL, EPSILON)

    assert is_verified, "Small random abstraction should likely remain robust."
    print("[PASS] Small random abstraction correctly kept robustness.")

# abstracting all but last layer


def test_all_except_last_layer_abstraction(label):
    print("\nRunning test_verification_with_wrong_label_fails...")
    verifier, input_image, initial_nap = load_test_objects(label)

    
    t=[0,1,2]
    for i in t:
        for j in range(len(initial_nap[i])):
                    initial_nap[i][j]=-1

    label=LABEL
    is_verified = verifier.is_verified_nap(initial_nap, input_image, label, EPSILON)

    assert is_verified, "Verification with evrything  abstracted exept last layer success."
    print("[PASS] Verification with with abstracted  layers except last one .")

def nap_size(nap):
    count = 0
    for i in range(len(nap)):
        for j in range(len(nap[i])):
            if nap[i][j] != -1:
                count += 1
    return count
def nap_last_layer_label(label):
    verifier, input_image, initial_nap = load_test_objects(label)

    
    t=[0,1,2]
    for i in t:
        for j in range(len(initial_nap[i])):
                    initial_nap[i][j]=-1

    label=LABEL
    is_verified = verifier.is_verified_nap(initial_nap, input_image, label, EPSILON)

    assert is_verified, "Verification with evrything  abstracted exept last layer success."
    return initial_nap
"""
def verify_coverage_nap(label,nap,input_image,model):
    
    label_to_samples = load_mnist_samples()
    labelNaps = []
    samples = label_to_samples[label]
    # Verify NAP coverage
    print("\n[INFO] Vérification de la couverture NAP :")
    if nap is None:
            print(f"Label {label}: NAP not defined.")
            


    correct = sum(follows_nap_array(x, model, nap) for x in samples)

        
    print(f"Label {label}: {correct}/{len(test_samples)} matched | {confused} confused samples from other labels")
        


"""


# then coarsen it 
if __name__ == "__main__":
    print("==== Starting NAP Verification Tests ====")
    #test_full_abstraction_fails()
    #test_initial_nap_is_robust()
    #test_verification_with_wrong_label_fails()
    #test_small_random_abstraction_should_remain_robust()
    label=1
    verifier, input_image, initial_nap = load_test_objects(label)

    mini_coarsen_label=nap_last_layer_label(label)
    #test_all_except_last_layer_abstraction(label)
    coarsened=stochasticShorten(mini_coarsen_label,input_image,label,-1,verifier,theta=0.1,max_iterations=10)
    diff = diff_naps(mini_coarsen_label, coarsened)
    print(f"[INFO] Coarsened neurons number: {diff}")
    print(f"[INFO] Coarsened nap size number: {nap_size(coarsened)}")


    # it passed 
    # so Now I am going to coarsen according to those
    # save it 
    nap_filename = f"coarsened_nap_label_minimal{label}.json"
    save_nap_to_json(coarsened, nap_filename)
    print("\n==== All NAP Verification Tests Passed ====")
