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
from src.utils.ntk_pytorch.ntk import get_ntk, get_ntk_eigenvalues
from src.utils.miscellaneous import nll_accuracy
import torch.nn.functional as F

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: OmegaConf) -> None:
    """
    Main function to run the basic NTK tracking experiment.
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
    train_set, test_set = dataset_factory(cfg.data, transform=transform, with_testset=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Setup model
    cfg.net.netparams.input_height = train_set[0][0].shape[1]
    cfg.net.netparams.input_width = train_set[0][0].shape[2]
    net = model_factory(cfg.net)
    net.to(device)

    # Setup learner
    learner = create_learner(cfg.learner, net, cfg.net)

    # Training loop
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs}')
        for input, target in pbar:
            input, target = input.to(device), target.to(device)
            loss, _ = learner.learn(input, target)
            pbar.set_postfix(loss=loss)

        # Evaluation
        if epoch % cfg.evaluation.eval_freq_epoch == 0:
            net.eval()
            test_loss = 0
            test_acc = 0
            with torch.no_grad():
                for input, target in test_loader:
                    input, target = input.to(device), target.to(device)
                    output = net(input)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    test_acc += nll_accuracy(output, target) * input.size(0)

            test_loss /= len(test_loader.dataset)
            test_acc /= len(test_loader.dataset)

            if cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'test_loss': test_loss,
                    'test_accuracy': test_acc
                })
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            net.train()

        # NTK Tracking
        if cfg.track_ntk and epoch % cfg.ntk_measure_freq_epoch == 0:
            print(f"Epoch {epoch+1}, computing NTK...")
            # Get a batch of data for NTK computation
            ntk_data_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.ntk_batch_size, shuffle=True)
            x_ntk, _ = next(iter(ntk_data_loader))
            x_ntk = x_ntk.to(device)

            # Compute NTK matrix
            ntk_matrix = get_ntk(net, x_ntk)

            # Compute eigenvalues
            eigenvalues = get_ntk_eigenvalues(ntk_matrix)

            # Log eigenvalues to wandb
            if cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'ntk_eigenvalues': wandb.Histogram(eigenvalues.cpu().numpy())
                })
            print("NTK computation complete.")

if __name__ == "__main__":
    main()
