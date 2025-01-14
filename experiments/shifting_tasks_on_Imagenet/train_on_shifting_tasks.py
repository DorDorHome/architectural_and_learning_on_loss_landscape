
"""

The most basic expriment to visualize the loss landscape.

"""



import random
from typing import Any
import sys
import pathlib

import test
# sys.path.append("../..")
# sys.path.append("../../..")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from configs.configurations import ExperimentConfig
# import json
import pickle
# import argparse
# import numpy as np
from tqdm import tqdm
import hydra 
import numpy as np


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
import torch.nn as nn
import os

# Function to initialize model weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
# should be moved to util:
def compute_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)
        correct = predicted.eq(target).sum().item()
        total = target.size(0)
        return correct / total

@hydra.main(config_path="cfg", config_name="shifting_tasks_config.yaml")
def main(cfg :ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    
    #setup network architecture, 
    
    #net = ConvNet(cfg.net.netparams)
    
    assert cfg.net.netparams.num_classes == cfg.num_classes_per_task
    
    net = model_factory(cfg.net)
    # set device for net
    net.to(cfg.device)
    
    # setup learner:
    # optimizer is setup in the learner
    # loss function is setup in the learner
    if cfg.learner.type == 'backprop':
        learner = Backprop(net, cfg.learner)
    if cfg.learner.type == 'cbp' and cfg.net.type == 'ConvNet':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet
        learner = ContinuousBackprop_for_ConvNet(net, cfg.learner)
    
        
    class_order_path = os.path.join(cfg.data.data_path, cfg.data.dataset, 'data','class_order')
    # load the class order:
    with open(class_order_path, 'rb+') as f:
        class_order = pickle.load(f)
        # class order is a numpy array of shape (300, 14000), containing number from 0 to 999
    
    # set class_order based on the run_id:
    class_order = class_order[cfg.run_id]
    
    # for extending the class_order to cover the tasks required:
    num_class_repetitions_required = int(cfg.num_classes_per_task*cfg.num_tasks /cfg.data.num_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)
    
    # setup data:
    transform = None
    Imagnet_dataset_generator = dataset_factory(cfg.data, transform)
    
    # for setting up batch:
    train_examples_per_epoch = cfg.train_size_per_class*cfg.num_classes_per_task
    
    epochs_per_task = cfg.epochs
    
    # wandb setup
    if cfg.use_wandb:
        import wandb
        print('finished importing wandb')
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.project_name, config= cfg_dict )

    
    
    
    for task_idx in tqdm(range(cfg.num_tasks)):
        # get the classes for the task:
        classes = class_order[task_idx*cfg.num_classes_per_task: (task_idx+1)*cfg.num_classes_per_task]
        
        # get the dataset for the classes:
        dataset_factory_by_classes = Imagnet_dataset_generator(train_images_per_class = cfg.train_size_per_class,
                                                                  test_images_per_class = cfg.val_size_per_class,
                                                                  test = cfg.test)

        # randomize internally:
        x_train, y_train, x_test, y_test = dataset_factory_by_classes(classes = classes, randomize = True)
        # put to device:        
        x_train, y_train, x_test, y_test = x_train.to(cfg.device), y_train.to(cfg.device), x_test.to(cfg.device), y_test.to(cfg.device)
        if cfg.new_heads_for_new_task:
            nn.init.kaiming_normal_(net.layers[-1].weight)
            net.layers[-1].bias.data *= 0
        
        # train the network on the task:
        for epoch in range(epochs_per_task):
            
            running_loss = 0.0
            #alternative way to track loss:
            batch_running_loss = 0.0
            
            # for accuracy
            number_of_correct = 0
            number_of_correct_2 = 0
            total = 0


            
            for batch_idx in range(0, train_examples_per_epoch, cfg.batch_size):
                x_batch = x_train[batch_idx: batch_idx+cfg.batch_size]
                y_batch = y_train[batch_idx: batch_idx+cfg.batch_size]
                loss, output = learner.learn(x_batch, y_batch)
                with torch.no_grad():
                    running_loss+= loss*x_batch.size(0)
                    batch_running_loss += loss
                    _, predicted = output.max(1)
                    total += y_batch.size(0)
                    number_of_correct += predicted.eq(y_batch).sum().cpu().item()
                    acc_batch = compute_accuracy(output, y_batch)
                    number_of_correct_2 += acc_batch * y_batch.size(0)
                    
                    
            
            if epoch % cfg.evaluation.eval_freq_epoch == 0:
                acc = number_of_correct/total
                acc_2 = number_of_correct_2/total
                loss = running_loss/total
                loss_by_batch = batch_running_loss/(train_examples_per_epoch/cfg.batch_size)

                data = {'task_idx*epochs_per_task + epoch': task_idx*epochs_per_task + epoch, 'train_accuracy': acc, 'train_accuracy_2': acc_2, 'train_loss': loss, 'train_loss_by_batch': loss_by_batch}
                # log to wandb:
                if cfg.use_wandb:
                    wandb.log(data)
                    
                # evaluate on test set:
                if cfg.evaluation.use_test_set:
                    with torch.no_grad():
                
                        number_of_correct_on_test = 0.0
                        number_of_correct_on_test_by_batch = 0.0
                        # test_running_loss = 0.0 
                        # test_running_loss_batch = 0.0
                        total = x_test.size(0)
                        for batch_idx in range(0, x_test.size(0), cfg.batch_size):
                            x_test_batch = x_test[batch_idx: batch_idx+cfg.batch_size]
                            y_test_batch = y_test[batch_idx: batch_idx+cfg.batch_size]
                            
                            test_output, _ = net.predict(x_test_batch)
                            
                            # test_running_loss +=
                            
                            # test_loss_batch = F.cross_entropy(test_output, y_test_batch)
                            # test_running_loss_batch += test_loss_batch*x_test_batch.size(0)
                            # one way to keep track of number of correct predictions:
                            _, predicted = test_output.max(1)
                            number_of_correct_on_test += predicted.eq(y_test_batch).sum().cpu().item()
                            # another way to keep track of number of correct predictions:
                            acc_batch_on_test = compute_accuracy(test_output, y_test_batch)
                            number_of_correct_on_test_by_batch += acc_batch_on_test * y_test_batch.size(0)

                        acc_on_test = number_of_correct_on_test/total
                        acc_on_test_2 = number_of_correct_on_test_by_batch/total
                    
                        data_on_test = {'task_idx*epochs_per_task + epoch': task_idx*epochs_per_task + epoch, 'test_accuracy': acc_on_test, 'test_accuracy_2': acc_on_test_2}
                    print(f"task: {task_idx}, Epoch: {epoch}, Accuracy: {acc}, Accuracy_2: {acc_2}, Loss:,  {loss}, loss by batch: {loss_by_batch}")
                    print(f"Test Accuracy: {acc_on_test}, Test Accuracy_2: {acc_on_test_2}")
                    if cfg.use_wandb:
                        wandb.log(data_on_test)
                        
        if task_idx % cfg.log_freq_every_n_task == 0:
            data_on_task = {'task': task_idx,\
                            'task_train_accuracy_after_all_epochs': acc,\
                            'task_train_accuracy_2_after_all_epochs': acc_2,
                            'task_train_loss_after_all_epochs': loss,
                            'task_train_loss_by_batch_after_all_epochs': loss_by_batch,
                            'task_test_accuracy_after_all_epochs': acc_on_test,
                            'task_test_accuracy_2_after_all_epochs': acc_on_test_2,}
            
            if cfg.use_wandb:
                wandb.log(data_on_task)

if __name__ == '__main__':
    main()