import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Learner(ABC):
    """
    Abstract base class for different learning algorithms.
    """

    def __init__(
        self,
        net: nn.Module,
        step_size: float = 0.001,
        loss: str = 'mse',
        opt: str = 'sgd',
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        weight_decay: float = 0.0,
        device: str = 'cpu',
        momentum: float = 0.0,
    ):
        """
        Initialize the Learner with network, optimizer, and loss function.

        :param net: Neural network model.
        :param step_size: Learning rate.
        :param loss: Loss function identifier ('mse' or 'nll').
        :param opt: Optimizer type ('sgd', 'adam', 'adamW').
        :param beta_1: Beta parameter for Adam optimizers.
        :param beta_2: Beta parameter for Adam optimizers.
        :param weight_decay: Weight decay (L2 regularization).
        :param device: Computation device ('cpu' or 'cuda').
        :param momentum: Momentum factor for SGD.
        """
        self.net = net.to(device)
        self.device = device

        # Initialize the optimizer
        self.opt = self._init_optimizer(opt, step_size, beta_1, beta_2, weight_decay, momentum)

        # Initialize the loss function
        self.loss = loss
        self.loss_func = self._init_loss_func(loss)

        # Placeholder for features or other tracking variables
        self.previous_features = None

    def _init_optimizer(
        self,
        opt: str,
        step_size: float,
        beta_1: float,
        beta_2: float,
        weight_decay: float,
        momentum: float
    ):
        """
        Initialize the optimizer based on the provided type.

        :return: Optimizer instance.
        """
        if opt == 'sgd':
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=step_size,
                weight_decay=weight_decay,
                momentum=momentum
            )
        elif opt == 'adam':
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=step_size,
                betas=(beta_1, beta_2),
                weight_decay=weight_decay
            )
        elif opt == 'adamW':
            optimizer = optim.AdamW(
                self.net.parameters(),
                lr=step_size,
                betas=(beta_1, beta_2),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt}")
        return optimizer

    def _init_loss_func(self, loss: str):
        """
        Initialize the loss function based on the provided identifier.

        :return: Loss function.
        """
        loss_funcs = {
            'nll': F.cross_entropy,
            'mse': F.mse_loss
        }
        if loss not in loss_funcs:
            raise ValueError(f"Unsupported loss type: {loss}")
        return loss_funcs[loss]

    @abstractmethod
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """
        Perform a learning step. Must be implemented by subclasses.

        :param x: Input tensor.
        :param target: Target tensor.
        :return: Loss and any additional metrics.
        """
        pass

    def _forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output and features.
        """
        return self.net.predict(x=x)
    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Learner(ABC):
    """
    Abstract base class for different learning algorithms.
    """

    def __init__(self, net: nn.Module, config: LearnerConfig):
        """
        Initialize the Learner with network, optimizer, and loss function.

        :param net: Neural network model.
        :param config: Learner configuration object.
        """
        self.net = net.to(config.device)
        self.device = config.device

        # Initialize the optimizer
        self.opt = self._init_optimizer(config)

        # Initialize the loss function
        self.loss = config.loss
        self.loss_func = self._init_loss_func(config.loss)

        # Placeholder for features or other tracking variables
        self.previous_features = None

    def _init_optimizer(self, config: LearnerConfig):
        """
        Initialize the optimizer based on the provided type.

        :return: Optimizer instance.
        """
        if config.opt == 'sgd':
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=config.step_size,
                weight_decay=config.weight_decay,
                momentum=config.momentum
            )
        elif config.opt == 'adam':
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=config.weight_decay
            )
        elif config.opt == 'adamW':
            optimizer = optim.AdamW(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.opt}")
        return optimizer

    def _init_loss_func(self, loss: str):
        """
        Initialize the loss function based on the provided identifier.

        :return: Loss function.
        """
        loss_funcs = {
            'nll': F.cross_entropy,
            'mse': F.mse_loss
        }
        if loss not in loss_funcs:
            raise ValueError(f"Unsupported loss type: {loss}")
        return loss_funcs[loss]

    @abstractmethod
    def learn(self, x: torch.Tensor, target: torch.Tensor):
        """
        Perform a learning step. Must be implemented by subclasses.

        :param x: Input tensor.
        :param target: Target tensor.
        :return: Loss and any additional metrics.
        """
        pass

    def _forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output and features.
        """
        return self.net.predict(x=x)