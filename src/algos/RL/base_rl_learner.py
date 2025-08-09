import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import gymnasium as gym

class BaseRLLearner(ABC):
    """
    Abstract base class for different reinforcement learning algorithms.
    This class is a wrapper around stable-baselines3 models.
    """

    def __init__(self, env: gym.Env, config: DictConfig, model_class=None):
        """
        Handle the setup of the RL agent.
        :param env: The gym environment to learn from.
        :param config: The configuration object for the learner.
        :param model_class: The stable-baselines3 model class (e.g., PPO, A2C).
        """
        self.env = env
        self.config = config
        self.device = config.device
        self.model_class = model_class
        self.model = self._create_model()

    def _create_model(self):
        """
        Create the RL model using the specified model_class and config.
        """
        if self.model_class is None:
            raise ValueError("model_class must be specified to create an RL model.")

        # Extract model-specific hyperparameters from the config
        model_kwargs = self.config.get('model_kwargs', {})

        # The policy network can be specified in the config.
        # If a custom policy is defined in the model factory, it will be used.
        # Otherwise, the default from stable-baselines3 is used.
        policy = self.config.get('policy', 'MlpPolicy')

        # Handle policy_kwargs for custom policies
        policy_kwargs = model_kwargs.pop('policy_kwargs', None)

        return self.model_class(
            policy=policy,
            env=self.env,
            device=self.device,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )

    def learn(self, total_timesteps: int, callback=None):
        """
        Train the RL agent for a given number of timesteps.
        """
        return self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, obs, deterministic: bool = True):
        """
        Predict the action for a given observation.
        """
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        """
        Save the trained model.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load a trained model.
        """
        # The model needs to be loaded with the specific class
        self.model = self.model_class.load(path, env=self.env)
        return self
