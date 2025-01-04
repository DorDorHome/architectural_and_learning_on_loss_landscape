# to-do

# requirements of model form:

The difference between the standard pytorch module and the model here is that:

1. Ouput of intermediate layers are often needed. So output of the forward method is of the form final_output, features, where features is a list of all the intermediate output.


# more complicated requirements:

For some applications, forward and backward hook would be required.