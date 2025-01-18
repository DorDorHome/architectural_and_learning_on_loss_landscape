# to-do
### add:
option not to use fan_in correction in svd_decomposed_FC.py and normalized weights_FC.py


# suitability of different models to dataset:

conv_net.ConvNet class is designed for image input of size [3, 32, 32]. Different input image size would need to be resized.

VGG in general can handle input shape [3, 224, 224]

# requirements of model form:

The difference between the standard pytorch module and the model here is that:

1. Ouput of intermediate layers are often needed. So output of the forward method is of the form final_output, features, where features is a list of all the intermediate output.


# more complicated requirements:

For some applications, forward and backward hook would be required.