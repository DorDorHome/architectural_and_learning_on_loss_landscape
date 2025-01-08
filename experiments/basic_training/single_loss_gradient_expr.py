
"""

The most basic expriment to visualize the loss landscape.

"""




import sys
import pathlib
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import hydra 
from omegaconf import DictConfig, OmegaConf
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
# import model factory:

from src.models.factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
import torchvision.transforms as transforms
import torchvision
import torch
# from dataclasses import asdict

@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    #setup network architecture, 
    
    # net = ConvNet(cfg.net.netparams)
    
    net = model_factory(cfg.net)
    
    if cfg.net.type == 'vgg_custom':
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images for the model
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        

        
    
    # setup learner:
    # optimizer is setup in the learner
    # loss function is setup in the learner
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
        
    # setup data:
    #load the data using data_factory:
    trainloader = dataset_factory(cfg.data)
    
    # if data_path is provided, use that.
    # if cfg.data.data_path is not None:
    #     dataset_path = cfg.data.data_path
    #     # for CIFAR10:
    #     if cfg.data.dataset == 'CIFAR10':
    #             # Define transformations for the training and test sets

    #         trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        
    #     # for mnist:
    #     elif cfg.data.dataset == 'MNIST':
    #         trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

    #     elif cfg.data.dataset == 'IMAGENET':
    #         trainset = torchvision.datasets.ImageNet(root=dataset_path, train=True, download=True, transform=transform)
            
    #     else:
    #         # raise error not implemented
    #         raise NotImplementedError("dataset not implemented")
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= cfg.batch_size, shuffle=True, num_workers=2, pin_memory = True)
    
    # load the data from the provided path:
    # else: 
    #     # raise an error if the data path is not provided
    #     raise ValueError("Data path is not provided.")

        # trainset = np.load(dataset_path)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size = cfg.batch_size, shuffle=True, num_workers=2, pin_memory = True)
        


    
    # setup evaluation:
    # wandb setup
    if cfg.use_wandb:
        import wandb
        print('finished importing wandb')
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project='basic_training', config= cfg_dict )

    
    
    
    if "accuracy" in cfg.evaluation.eval_metrics:
        accuracy = nll_accuracy
    if 'loss' in cfg.evaluation.eval_metrics:
        loss_func  = F.cross_entropy
        
    # training loop:
    # set net to training mode
    net.train()
    for epoch in tqdm(range(cfg.epochs), desc='Epoch'):
        
        # running loss for the epoch:
        running_loss = 0.0
        
        # for accuracy
        number_of_correct = 0
        total = 0
        
        for input, label in tqdm(trainloader, desc='Batch'):
            loss, output = learner.learn(input, label)
            running_loss += loss*input.size(0)
            _, predicted = output.max(1)
            total += label.size(0)
            predicted = predicted.cpu()
            number_of_correct += predicted.eq(label).sum().item()
            #torch.max(output.data, 1)

        # evaluate:
        if epoch % cfg.evaluation.eval_freq_epoch == 0:
            acc = number_of_correct/total
            loss = running_loss/total
            # with torch.no_grad():
            #     y_pred, _ = net.predict(x)
            #     acc = accuracy(y_pred, y)
            #     loss = loss_func(y_pred, y)
            print(f"Epoch: {epoch}, Accuracy: {acc}, Loss: {loss}")
            
            data = {'epoch': epoch, 'accuracy': acc, 'loss': loss}
            # log to wandb:
            if cfg.use_wandb:
                wandb.log(data)
                

        
    

if __name__ == "__main__":
    main()
    
    