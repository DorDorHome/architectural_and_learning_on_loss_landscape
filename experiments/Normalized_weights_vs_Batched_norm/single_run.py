from typing import Any
import sys
import pathlib
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from configs.configurations import ExperimentConfig

import os
# import pickle
# import argparse
# import numpy as np
from tqdm import tqdm
import hydra 
from omegaconf import OmegaConf # , DictConfig
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
# import model factory:
from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
import torchvision.transforms as transforms
import torchvision
import torch

@hydra.main(config_path="cfg", config_name="config_for_linear_type")
def main(cfg: ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    
    net = model_factory(cfg.net)
    
    # verify cfg.runs == 1:
    if cfg.runs != 1:
        raise ValueError("This script is only for single runs. Please set runs to 1 or use another script.")

    # if the run_id attribute is not set, set it to 0
    if cfg.run_id is None:
        cfg.run_id = 0
        
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
    elif cfg.learner.type == 'cbp' and cfg.net.type == 'conv_net':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinualBackprop_for_ConvNet
        learner = ContinualBackprop_for_ConvNet(net, cfg.learner)
    
    # setup data:

    transform = transform_factory(cfg.data.dataset, cfg.net.type)

        
    trainset, testset = dataset_factory(cfg.data, transform, with_testset=cfg.evaluation.use_testset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    if cfg.evaluation.use_testset:
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
    if cfg.use_wandb:
        import wandb
        print('finished importing wandb')
        try:
            from src.utils.task_shift_logging import build_logging_config_dict
            cfg_dict = build_logging_config_dict(cfg)
        except Exception as e:
            print(f"Warning: task shift logging sanitization failed, using full config. Error: {e}")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, config=cfg_dict)

    if "accuracy" in cfg.evaluation.eval_metrics:
        accuracy = nll_accuracy
    if 'loss' in cfg.evaluation.eval_metrics:
        loss_func = F.cross_entropy
        
    net.train()
    for epoch in tqdm(range(cfg.epochs), desc='Epoch'):
        running_loss = 0.0
        number_of_correct = 0
        total = 0
        for input, label in tqdm(trainloader, desc=f"Epoch: {epoch}, progress on batches", leave=True):
            input, label = input.to(cfg.device), label.to(cfg.device)
            if cfg.net.network_class == 'fc':
                input = input.flatten(start_dim=1)
                
            loss, output = learner.learn(input, label)
            running_loss += loss * input.size(0)
            _, predicted = torch.max(output, 1)
            number_of_correct += (predicted == label).sum().cpu().item()
            total += label.size(0)
                
        if epoch % cfg.evaluation.eval_freq_epoch == 0:
            acc = number_of_correct / total
            loss = running_loss / total
            print(f"Epoch: {epoch}, trainset accuracy: {acc}, trainset loss: {loss}")
            data = {'epoch': epoch, 'trainset_accuracy': acc, 'trainset_loss': loss}
            if cfg.use_wandb:
                print(f"Logging to WandB: {data}")
                wandb.log(data)
                    
        if cfg.evaluation.use_testset and testset is not None:
            net.eval()
            number_of_correct_test = 0
            running_loss_test = 0.0
            total_test = 0
            for input, label in testloader:
                input, label = input.to(cfg.device), label.to(cfg.device)
                if cfg.net.network_class == 'fc':
                    input = input.flatten(start_dim=1)
                with torch.no_grad():
                    output, _ = net.predict(input)
                    loss = loss_func(output, label)
                    running_loss_test += loss * input.size(0)
                    _, predicted = torch.max(output, 1)
                    number_of_correct_test += (predicted == label).sum().cpu().item()
                    total_test += label.size(0)
            acc_test = number_of_correct_test / total_test
            loss_test = running_loss_test / total_test
            print(f"Epoch: {epoch}, testset accuracy: {acc_test}, testset loss: {loss_test}")
            data = {'epoch': epoch, 'testset_accuracy': acc_test, 'testset_loss': loss_test}
            if cfg.use_wandb:
                wandb.log(data)


if __name__ == "__main__":  
    main()








