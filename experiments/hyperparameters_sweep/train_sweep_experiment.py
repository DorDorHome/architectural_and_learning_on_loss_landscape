from __future__ import annotations

import random
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(params, cfg: DictConfig):
    if cfg.learner.opt == 'sgd':
        return torch.optim.SGD(params, lr=cfg.learner.step_size, momentum=cfg.learner.momentum, weight_decay=cfg.learner.weight_decay)
    elif cfg.learner.opt == 'adam':
        return torch.optim.Adam(params, lr=cfg.learner.step_size, weight_decay=cfg.learner.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.learner.opt}")


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Any:
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {cfg.seed}")
    set_seed(int(cfg.seed))

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(cfg.data.data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=int(cfg.num_workers), pin_memory=True)

    # Model
    model = SimpleCNN(num_classes=int(cfg.data.num_classes)).to(device)
    optimizer = build_optimizer(model.parameters(), cfg)
    criterion = nn.CrossEntropyLoss()

    if cfg.get('use_wandb', False):
        import wandb
        try:
            from src.utils.task_shift_logging import build_logging_config_dict
            cfg_dict = build_logging_config_dict(cfg)
        except Exception as e:
            print(f"Warning: task shift logging sanitization failed, using full config. Error: {e}")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, config=cfg_dict)

    model.train()
    for epoch in range(int(cfg.epochs)):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_seen += int(inputs.size(0))

        epoch_loss = total_loss / max(total_seen, 1)
        epoch_acc = total_correct / max(total_seen, 1)

        if cfg.get('use_wandb', False):
            import wandb
            wandb.log({"epoch": epoch, "epoch_loss": epoch_loss, "epoch_accuracy": epoch_acc})

    print(f"METRIC epoch_loss={epoch_loss}")
    print(f"METRIC epoch_accuracy={epoch_acc}")
    return 0


if __name__ == "__main__":
    main()
