"""
The basic experiment, with rank tracking.


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

# rank tracking:
from src.utils.miscellaneous import compute_matrix_rank_summaries
from src.utils.forward_extraction_tools_source import compute_rank_for_list_of_features

import torchvision.transforms as transforms
import torchvision
import torch



@hydra.main(config_path="cfg", config_name="rank_tracking_config")
def main(cfg: ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    
    #setup network architecture, 
    
    # net = ConvNet(cfg.net.netparams)
    
    net = model_factory(cfg.net)

    #verifty cfg.runs ==1:
    if cfg.runs != 1:
        raise ValueError("This script is only for single runs. Please set runs to 1 or use another script.")

    # if the run_id attribute is not set, set it to 0
    if cfg.run_id is None:
        cfg.run_id = 0
        
    # setup learner:
    # optimizer is setup in the learner
    # loss function is setup in the learner
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
    if cfg.learner.type == 'cbp' and cfg.net.type == 'conv_net':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinualBackprop_for_ConvNet
        learner = ContinualBackprop_for_ConvNet(net, cfg.learner)
    
    # setup data:
    # load the transfrom based on the dataset and model:
    # combination of dataset and model determines the transform
    transform = transform_factory(cfg.data.dataset, cfg.net.type)

    #trainset with the transform:
    trainset, _ = dataset_factory(cfg.data, transform )

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
        wandb.init(project=cfg.wandb.project, config= cfg_dict )

    if cfg.use_json:
        from src.utils.data_logging import save_data_json
        import json
        # set up the folder for the json file:
        # find a exp_id, integer, that is not used in the experiments folder:
        exp_id = 0
        dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments',
                                          'basic_training', f'experiment_cfg_{exp_id}')
        
        run_dir = os.path.join(dir_for_experiment, 'run', f'run_{cfg.run_id}')
        
        while os.path.exists(dir_for_experiment):
            exp_id += 1
            dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments', 'basic_training', f'experiment_cfg_{exp_id}')
            os.makedirs(dir_for_experiment, exist_ok=True)
        
        
        # create a file to save the in dir_for_experiment, containing the config
        os.makedirs(dir_for_experiment, exist_ok=True)
        with open(os.path.join(dir_for_experiment, 'config.json'), 'w') as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)




        # create a folder for the run:
        os.makedirs(run_dir, exist_ok=True)
    
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
            if input is None or label is None:
                print("Found None in the data loader batch")
                continue
            
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
        
\
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
            
            # extrack rank:
            if epoch % cfg.rank_measure_freq_to_epoch == 0:
                # features:
                list_of_features_for_every_layers = learner.previous_features
                
                
                rank_summary_list = compute_rank_for_list_of_features(
                    list_of_features=list_of_features_for_every_layers,
                    extraction_function=compute_matrix_rank_summaries)
                
                for layer_idx in range(len(list_of_features_for_every_layers)):
                    data[f'layer_{layer_idx}_rank'] = rank_summary_list[layer_idx][0]
                    data[f'layer_{layer_idx}_effective_rank'] = rank_summary_list[layer_idx][1]
                    data[f'layer_{layer_idx}_approximate_rank'] = rank_summary_list[layer_idx][2]
                    data[f'layer_{layer_idx}_approximate_rank_abs'] = rank_summary_list[layer_idx][3]    
        
        
            
            # log to wandb:
            if cfg.use_wandb:
                wandb.log(data)
            if cfg.use_json:
                save_data_json(data, run_dir, filename=f'run_{cfg.run_id}.json')

                

        
    

if __name__ == "__main__":
    # clear cached memory:
    torch.cuda.empty_cache()
    
    
    main()
    
    
    

