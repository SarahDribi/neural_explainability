This repository is made to provide neural based explanations for mnist pictures , tabular data in 
general, particularly for classification  models built exclusively with relu activations.
The repository is structured like this:
****Models****
Some onnx classification models that are supported by this framework
*****Tools******
******Plnn*******
This repo relies on the oval-bab framework, using many functions to verify some robustness 
properties at an input region 
******The clean directory*******
This directory has the following directories:
**********nap_extraction dir*********
This module is responsible of extracting the set of activation status , given a 
supported classification model and an input as a tensor , it returns 
a represention of the activation statuses that occured when the input was passed on the model :
If the model has 2 layers , if the first layer has 3 neurons and the second has 2 neurons,
if it returns [[1,0,1],[1,0]] this means that first neuron in first layer was active , so was the last 
the last neuron in the first layer , so was the first neuron in the second layer , the rest were inactive.
-

