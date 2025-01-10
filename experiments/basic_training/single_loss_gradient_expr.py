
"""

The most basic expriment to visualize the loss landscape.

"""



from typing import Any
import sys
import pathlib
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from configs.configurations import ExperimentConfig
# import json
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


# should be moved to util:
def compute_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)
        correct = predicted.eq(target).sum().item()
        total = target.size(0)
        return correct / total




@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg :ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    
    #setup network architecture, 
    
    # net = ConvNet(cfg.net.netparams)
    
    net = model_factory(cfg.net)


    # setup learner:
    # optimizer is setup in the learner
    # loss function is setup in the learner
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
        
    # setup data:
    # load the transfrom based on the dataset and model:
    transform = transform_factory(cfg.data.dataset, cfg.net.type)

    #trainset with the transform:
    trainset = dataset_factory(cfg.data, transform )

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

        # alternative way to track loss, by batch loss
        batch_running_loss = 0.0
        
        # for accuracy
        number_of_correct = 0
        number_of_correct_2 =0
        total = 0

        
        for input, label in tqdm(trainloader, desc=f"Epoch: {epoch}, progress on batches", leave =True):
            input, label = input.to(cfg.device), label.to(cfg.device)
            
            # print(label.dtype)  # Should be torch.long for CrossEntropyLoss
            # print(label.min(), label.max())  # Should be within [0, num_classes - 1]
            loss, output = learner.learn(input, label)
            running_loss+= loss*input.size(0)
            batch_running_loss += loss
            _, predicted = output.max(1)
            total += label.size(0)
            # predicted = predicted.cpu()
            number_of_correct += predicted.eq(label).sum().cpu().item()
            #torch.max(output.data, 1)
            acc_batch = compute_accuracy(output, label)
            number_of_correct_2 += acc_batch * label.size(0)
        # evaluate:
        if epoch % cfg.evaluation.eval_freq_epoch == 0:
            acc = number_of_correct/total
            acc_2 = number_of_correct_2/total
            loss = running_loss/total
            loss_by_batch = batch_running_loss/len(trainloader)
            # with torch.no_grad():
            #     y_pred, _ = net.predict(x)
            #     acc = accuracy(y_pred, y)
            #     loss = loss_func(y_pred, y)
            print(f"Epoch: {epoch}, Accuracy: {acc}, Accuracy_2: {acc_2}, Loss:,  {loss}, loss by batch: {loss_by_batch}")
            
            data = {'epoch': epoch, 'accuracy': acc, 'accuracy_2': acc_2, 'loss': loss, 'loss_by_batch': loss_by_batch}
            # log to wandb:
            if cfg.use_wandb:
                wandb.log(data)
                

        
    

if __name__ == "__main__":
    main()
    
    