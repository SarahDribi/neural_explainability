

# apply coarsening to image 

# the image is in pt    format
# nap is a list of lists in a json format
# the model is a onnx model
# the coarsening mode is entered 
# it could be stochastic or deterministic or heuristic
# it s a string

# apply the coarsening to the image 


# result is a dictionary of things that we could process with other routines
def image_minimal_nap(image_path,image_nap,onnx_model_path,coarsening_mode):

    # Process outputs based on coarsening mode
    result={}
    if coarsening_mode == 'stochastic':
        # Apply stochastic coarsening logic here
        pass
    elif coarsening_mode == 'deterministic':
        # Apply deterministic coarsening logic here
        pass
    elif coarsening_mode == 'heuristic':
        # Apply heuristic coarsening logic here
        pass

    return result

