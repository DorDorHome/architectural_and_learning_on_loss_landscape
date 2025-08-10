import warnings
warnings.filterwarnings("ignore", message="A NumPy version")
warnings.filterwarnings("ignore", message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

import sys
import pathlib
import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
from src.algos.supervised.supervised_factory import create_learner
from src.data_loading.shifting_dataset import create_stateful_dataset_wrapper, create_stateless_dataset_wrapper
from src.utils.zeroth_order_features import compute_all_rank_measures_list
from src.utils.rank_drop_dynamics import compute_rank_dynamics_from_features
from src.utils.ntk_pytorch.ntk import get_ntk, get_ntk_eigenvalues

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: OmegaConf) -> None:
    """
    Main function to run the plasticity and NTK tracking experiment.
    """
    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    # Setup wandb
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))

    # Setup device
    device = torch.device(cfg.device)

    # Setup data
    transform = transform_factory(cfg.data.dataset, cfg.net.type)
    train_set, _ = dataset_factory(cfg.data, transform=transform, with_testset=False)

    # Setup model
    cfg.net.netparams.input_height = train_set[0][0].shape[1]
    cfg.net.netparams.input_width = train_set[0][0].shape[2]
    net = model_factory(cfg.net)
    net.to(device)

    # Setup learner
    learner = create_learner(cfg.learner, net, cfg.net)

    is_stateful = cfg.task_shift_mode in ["drifting_values", 'continuous_input_deformation']
    dataset_wrapper = None
    if is_stateful:
        dataset_wrapper = create_stateful_dataset_wrapper(cfg, train_set)

    # Training loop
    for task_idx in range(cfg.num_tasks):
        print(f"\n{'='*50}")
        print(f"Starting task {task_idx+1}/{cfg.num_tasks}")
        print(f"{'='*50}")

        if is_stateful:
            if hasattr(dataset_wrapper, 'update_task'):
                dataset_wrapper.update_task()
            current_train_set = dataset_wrapper
        else:
            current_train_set = create_stateless_dataset_wrapper(cfg, train_set, task_idx)

        train_loader = torch.utils.data.DataLoader(current_train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        if hasattr(dataset_wrapper, 'update_drift'):
            print("Updating dataset drift...")
            dataset_wrapper.update_drift()

        pbar = tqdm(range(cfg.epochs), desc=f'Task {task_idx+1}/{cfg.num_tasks}')
        for epoch in pbar:
            for input, target in train_loader:
                input, target = input.to(device), target.to(device)
                loss, _ = learner.learn(input, target)
                pbar.set_postfix(loss=loss)

        # --- Metrics Tracking at Task Switch ---
        print(f"Task {task_idx+1} complete. Computing metrics...")

        data_log = {'task_idx': task_idx}

        # Plasticity / Rank Tracking
        if cfg.track_rank:
            # Get features
            specified_loader = torch.utils.data.DataLoader(current_train_set, batch_size=cfg.specified_batch_size, shuffle=True)
            x_rank, _ = next(iter(specified_loader))
            x_rank = x_rank.to(device)
            _, features = net.predict(x_rank)
            features = [f.view(f.size(0), -1) for f in features]

            # Compute rank measures
            rank_summary_list = compute_all_rank_measures_list(
                features=features,
                use_pytorch_entropy_for_effective_rank=cfg.use_pytorch_entropy_for_effective_rank,
                prop_for_approx_or_l1_rank=cfg.prop_for_approx_or_l1_rank,
                numerical_rank_epsilon=cfg.numerical_rank_epsilon
            )
            for i, summary in enumerate(rank_summary_list):
                for name, value in summary.items():
                    data_log[f'layer_{i}_{name}'] = value

            # Compute rank drop dynamics
            if cfg.track_rank_drop:
                rank_dynamics = compute_rank_dynamics_from_features(
                    feature_list=features,
                    rank_summary_list=rank_summary_list,
                    batch_size=cfg.specified_batch_size,
                    mode=cfg.rank_drop_mode,
                    use_theoretical_max_first=cfg.from_theoretical_max_first_feature_rank
                )
                data_log.update(rank_dynamics)

        # NTK Tracking
        if cfg.track_ntk:
            # Get NTK data
            ntk_loader = torch.utils.data.DataLoader(current_train_set, batch_size=cfg.ntk_batch_size, shuffle=True)
            x_ntk, _ = next(iter(ntk_loader))
            x_ntk = x_ntk.to(device)

            # Compute NTK matrix and eigenvalues
            ntk_matrix = get_ntk(net, x_ntk)
            eigenvalues = get_ntk_eigenvalues(ntk_matrix)

            data_log['ntk_eigenvalues'] = wandb.Histogram(eigenvalues.cpu().numpy()) if cfg.use_wandb else eigenvalues.cpu().numpy()

        # Log to wandb
        if cfg.use_wandb:
            wandb.log(data_log)

        print("Metrics computation complete.")


if __name__ == "__main__":
    main()
