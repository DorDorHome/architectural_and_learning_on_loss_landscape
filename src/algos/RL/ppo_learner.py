from src.algos.RL.base_rl_learner import BaseRLLearner
from stable_baselines3 import PPO
from omegaconf import DictConfig
import gymnasium as gym

class PPOLearner(BaseRLLearner):
    """
    A learner for the PPO algorithm from stable-baselines3.
    """
    def __init__(self, env: gym.Env, config: DictConfig):
        """
        Initialize the PPO learner.
        :param env: The gym environment to learn from.
        :param config: The configuration object for the learner.
        """
        super().__init__(env, config, model_class=PPO)
