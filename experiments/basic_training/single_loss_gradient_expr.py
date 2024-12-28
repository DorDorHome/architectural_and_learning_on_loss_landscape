
"""

The most basic expriment to visualize the loss landscape.

"""

import sys
sys.path.append("../..")
sys.path.append("../../..")
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import hydra 
from omegaconf import DictConfig, OmegaConf
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
# import network:
from src.models.conv_net import ConvNet
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy



@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    #setup network architecture:
    if cfg.net.type == 'ConvNet':
        net = ConvNet(cfg.net)
    
    # setup learner:
    # optimizer is setup in the learner
    # loss function is setup in the learner
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
        
    # setup data:
    with open(cfg.data_path, 'rb') as f:
        x, y, _, _ = pickle.load(f)
        
    # setup evaluation:
    if "accuracy" in cfg.evaluation.eval_metrics:
        accuracy = nll_accuracy
    if 'loss' in cfg.evaluation.eval_metrics:
        loss_func  = F.cross_entropy
        

        
    # training loop:
    for epoch in range(cfg.epochs):
        for i in range(0, x.shape[0], cfg.batch_size):
            x_batch = x[i:i+cfg.batch_size]
            y_batch = y[i:i+cfg.batch_size]
            learner.train(x_batch, y_batch)
            
        # evaluate:
        if epoch % cfg.eval_freq_epoch == 0:
            y_pred = learner.predict(x)
            acc = accuracy(y_pred, y)
            loss = loss_func(y_pred, y)
            print(f"Epoch: {epoch}, Accuracy: {acc}")

        
    

if __name__ == "__main__":
    main()
    
    