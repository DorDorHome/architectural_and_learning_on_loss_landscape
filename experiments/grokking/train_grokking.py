import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import random
import numpy as np
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.algos.supervised.supervised_factory import create_learner
from src.utils.zeroth_order_features import compute_all_rank_measures_list
from src.data_loading.transform_factory import transform_factory

def nll_accuracy(output, target):
    """Computes accuracy for classification tasks."""
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / len(target)

@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    # --- Setup ---
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device(cfg.device)

    # --- Data ---
    # The grokking dataset doesn't need a transform
    train_set, test_set = dataset_factory(cfg.data, transform=None, with_testset=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # --- Model and Learner ---
    net = model_factory(cfg.net).to(device)
    learner = create_learner(cfg.learner, net, cfg.net)

    # --- Training Loop ---
    pbar = tqdm(range(cfg.epochs))
    for epoch in pbar:
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            loss, output = learner.learn(input_batch, target_batch)
            train_loss += loss.item()
            train_acc += nll_accuracy(output[:, -1, :], target_batch)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        log_data = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

        # --- Evaluation ---
        if epoch % cfg.evaluation.eval_freq_epoch == 0:
            net.eval()
            test_loss = 0.0
            test_acc = 0.0
            with torch.no_grad():
                for input_batch, target_batch in test_loader:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                    output = net(input_batch)
                    # For grokking, we only care about the last output token for loss/accuracy
                    loss = torch.nn.functional.cross_entropy(output[:, -1, :], target_batch)
                    test_loss += loss.item()
                    test_acc += nll_accuracy(output[:, -1, :], target_batch)

            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            log_data.update({"test_loss": test_loss, "test_acc": test_acc})
            pbar.set_description(f"E: {epoch}, TrL: {train_loss:.3f}, TrA: {train_acc:.3f}, TeL: {test_loss:.3f}, TeA: {test_acc:.3f}")

            # --- Rank Measurement ---
            if cfg.track_rank:
                with torch.no_grad():
                    # Get features from the last training batch
                    list_of_features_for_every_layers = learner.previous_features
                    
                    list_of_features_for_every_layers = [f.view(f.size(0), -1) for f in list_of_features_for_every_layers]

                    rank_summary_list = compute_all_rank_measures_list(
                        features=list_of_features_for_every_layers
                    )
                    
                    for i, summary in enumerate(rank_summary_list):
                        layer_name = cfg.layers_identifier[i] if cfg.layers_identifier else f"layer_{i}"
                        for rank_type, value in summary.items():
                            log_data[f"{layer_name}_{rank_type}"] = value

        if cfg.use_wandb:
            wandb.log(log_data)

if __name__ == "__main__":
    main()
