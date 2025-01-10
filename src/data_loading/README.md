# folder of dataset:

It is recommended that dataset be read from the dataset path in the local computer, to avoid duplicating the data from different projects.

## options:
the DataConfig contains a option called use_torchvision. When set to False, custom dataset will be used.

## transform factory
the dataset factory depends on the transform_factory, which handles what transform to apply on dataset, based on the dataset used and the model used.

