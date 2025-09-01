"""



def sample_image_noise(input_image,epsilon_ball,noise_factor):

def diff(nap_one,nap_two): # returns a set of  neurons that have different activations
    return

# suppose that 
# float neurons are neurons that change activation in the area 

# We consider that the input_image is robust to adeversarial attacks
def get_important_neurons(input_image,input_label,epsilon_ball,noise_factor,model,max_iterations):
    nap_one=extract_activation_pattern(input_image,model)
    
    close_image=sample_image_noise(input_image,epsilon_ball,noise_factor)
        # compute the activation pattern 
    nap_two=extract_activation_pattern(close_image,model)
    prediction=model_prediction(model,close_image)
    difference=diff(nap_one,nap_two)
    #two cases 
    # if the decision changes within this Nap 
    # if the decision changes within this Neighboring Nap then we consider all the neurons as important
    # the intuition is that I should 


def get_important_neurons(input_image,input_label,epsilon):
    # same thing but doing the attacks 

        
        
        

"""
