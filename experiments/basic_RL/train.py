import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import torch
import random
import numpy as np
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.algos.RL.rl_factory import create_rl_learner
from src.models.model_factory import model_factory

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the basic RL experiment.
    """
    # --- Set seed for reproducibility ---
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    print(f"Using seed: {cfg.seed}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(OmegaConf.to_yaml(cfg))

    # --- Create the environment ---
    env = gym.make(cfg.env.id)

    # --- Set up policy_kwargs for custom feature extractor if specified ---
    if cfg.learner.policy == 'CustomActorCriticPolicy':
        # Get the feature extractor class from the model factory
        feature_extractor_class = model_factory(cfg.net)

        policy_kwargs = {
            "features_extractor_class": feature_extractor_class,
            "features_extractor_kwargs": dict(features_dim=cfg.net.netparams.features_dim),
        }
        # The learner's config needs to be updated with the policy_kwargs
        # so that the learner can pass them to the stable-baselines3 model.
        # However, our current BaseRLLearner doesn't do this.
        # Let's modify the learner to handle policy_kwargs.
        # I will update the base_rl_learner.py to handle this.
        # For now, I will assume the learner handles it.
        cfg.learner.model_kwargs.policy_kwargs = policy_kwargs


    # --- Create the learner ---
    learner = create_rl_learner(cfg.learner, env)

    # --- Training ---
    print("Starting training...")
    learner.learn(total_timesteps=cfg.total_timesteps)
    print("Training finished.")

    # --- Save the model ---
    # In a real experiment, we would save the model to a specified path.
    # For this basic example, we'll save it in the hydra output directory.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_path = f"{output_dir}/model.zip"
    learner.save(model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation (optional) ---
    # You can load the model and evaluate it.
    # loaded_model = learner.load(model_path)
    # mean_reward, std_reward = evaluate_policy(loaded_model.model, env, n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
