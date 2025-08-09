import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import torch
import random
import numpy as np
import sys
import pathlib
import wandb
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.algos.RL.rl_factory import create_rl_learner
from src.models.model_factory import model_factory
from src.utils.zeroth_order_features import compute_all_rank_measures_list

class RankTrackingCallback(BaseCallback):
    """
    A custom callback to track the rank of features in the policy network.
    """
    def __init__(self, track_freq: int, verbose=0):
        super(RankTrackingCallback, self).__init__(verbose)
        self.track_freq = track_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.track_freq == 0:
            # Get the policy network
            policy = self.model.policy

            # We need to get some observations to pass through the network.
            # We can get them from the replay buffer if it's available,
            # or from the environment.
            # For on-policy algorithms like PPO, the rollout buffer is available.
            obs_tensor = self.model.rollout_buffer.observations

            # The feature extractor is part of the policy
            features_extractor = policy.features_extractor

            # Pass the observations through the feature extractor to get features
            features = features_extractor(obs_tensor)

            # The features can be a single tensor or a dict of tensors.
            # Our simple backbones return a single tensor.

            # The rank computation function expects a list of feature tensors.
            # In our case, we have only one feature tensor from the backbone.
            # To track rank dynamics, we would need to extract features from
            # intermediate layers of the backbone, which requires modifying the backbones.
            # For now, we will just track the rank of the final feature output.

            list_of_features = [features.view(features.size(0), -1)]

            rank_summary_list = compute_all_rank_measures_list(features=list_of_features)

            # Log the rank metrics to wandb
            for i, rank_summary in enumerate(rank_summary_list):
                for key, value in rank_summary.items():
                    wandb.log({f"rank/layer_{i}/{key}": value}, step=self.num_timesteps)

            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}, rank metrics logged.")

        return True

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the tracking in RL experiment.
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

    # --- Set up wandb ---
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
        )

    # --- Create the environment ---
    env = gym.make(cfg.env.id)

    # --- Set up policy_kwargs for custom feature extractor ---
    if 'net' in cfg and cfg.net.type is not None:
        print("Using custom backbone.")
        feature_extractor_class = model_factory(cfg.net)

        policy_kwargs = {
            "features_extractor_class": feature_extractor_class,
            "features_extractor_kwargs": dict(cfg.net.netparams)
        }

        if 'model_kwargs' not in cfg.learner:
            cfg.learner.model_kwargs = {}
        cfg.learner.model_kwargs.policy_kwargs = policy_kwargs

    # --- Create the learner ---
    learner = create_rl_learner(cfg.learner, env)

    # --- Set up the callback ---
    callback = None
    if cfg.tracking.enabled:
        callback = RankTrackingCallback(track_freq=cfg.tracking.track_rank_freq)

    # --- Training ---
    print("Starting training...")
    learner.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    print("Training finished.")

    # --- Save the model ---
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_path = f"{output_dir}/model.zip"
    learner.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    if cfg.logging.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
