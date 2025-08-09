from omegaconf import DictConfig
import gym
from src.algos.RL.ppo_learner import PPOLearner

def create_rl_learner(config: DictConfig, env: gym.Env):
    """
    Factory function to create RL learner instances based on the configuration.

    Args:
        config (DictConfig): Configuration object containing learner parameters.
        env (gym.Env): The gym environment to learn from.

    Returns:
        An instance of the requested RL learner.

    Raises:
        ValueError: If the specified learner type is unsupported.
    """
    learner_type = config.type

    if learner_type == 'ppo':
        return PPOLearner(env, config)
    # Add other RL learners here as they are implemented
    # elif learner_type == 'a2c':
    #     return A2CLearner(env, config)
    else:
        raise ValueError(f"Unsupported RL learner type: {learner_type}")
