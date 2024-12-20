# Objectives for Learner:
# Initialization: Handle the setup of the neural network, optimizer, and loss function.
# Learning Process: Define a generic learn method that can be extended or overridden by subclasses.
# Utility Functions: Include any shared utility methods that multiple learners might use.

import torch
import torch.nn.functional as F
from torch import optim
from abc import ABC, abstractmethod

class Learner(ABC):
    """
    abstract base class for different learning algorithms
    """

    def __init__():

        # handle the setup of networks(agents), optimizer, and loss function.

    def _init_

    def learn()