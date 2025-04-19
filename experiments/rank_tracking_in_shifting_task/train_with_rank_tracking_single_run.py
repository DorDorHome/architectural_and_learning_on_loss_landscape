"""
shift tasks, with rank tracking.
Purpose:
to see the correlation (if any) between the loss of plasticity and the rank of the features, rank of CNN filters, dead units, rank of CNN filters times zerod activation during the training process.

"""

from typing import Any
import sys
import pathlib
import torch.utils

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
import numpy as np #used in concatenating the data
from omegaconf import OmegaConf # , DictConfig
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
# import model factory:

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
from src.data_loading.shifting_dataset import PermutedDataset
# rank tracking:
from src.utils.zeroth_order_features import compute_all_rank_measures_list, count_saturated_units_list
#from src.utils.miscellaneous import compute_matrix_rank_summaries
#from src.utils.partial_jacobian_source import compute_rank_for_list_of_features

# weight magnitude tracking:
from src.utils.track_weights_norm import track_weight_stats

import torchvision.transforms as transforms
import torchvision
import torch


from src.utils.miscellaneous import compute_accuracy

@hydra.main(config_path="cfg", config_name="rank_tracking_in_shifting_tasks_config_conv")
def main(cfg: ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))
    # set up the transform being used for the dataset, given the model architecture and dataset
    # expected cfg.data.dataset value is 'MNIST'
    # expected cfg.net.type value is 'ConvNet'
    transform = transform_factory(cfg.data.dataset, cfg.net.type)
    
    # setup data:
    train_set, _ = dataset_factory(cfg.data, transform = transform, with_testset= False)
    
    # for setting up batch:
    # train_examples_per_epoch = cfg.train_size_per_class*cfg.num_classes_per_task
     
    
    
    
    
    # Convert "None" string to Python None
    if cfg.layers_identifier == "None":
        cfg.layers_identifier = None
    
    # making sure the number of classes of the dataset matches the number of classes in the network:
    if cfg.net.network_class == 'fc':
        assert cfg.net.netparams.num_outputs == cfg.data.num_classes
    elif cfg.net.network_class == 'conv':
        assert cfg.net.netparams.num_classes == cfg.data.num_classes
    
    #extract dataset parameters for setting up model:
    cfg.net.netparams.input_height = train_set[0][0].shape[1]
    cfg.net.netparams.input_width = train_set[0][0].shape[2]
    #setup network architecture
    net = model_factory(cfg.net)
    # set device for net
    net.to(cfg.net.device)

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
        learner = Backprop(net, cfg.learner, cfg.net)
    if cfg.learner.type == 'cbp' and cfg.net.type == 'ConvNet':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinuousBackprop_for_ConvNet
        learner = ContinuousBackprop_for_ConvNet(net, cfg.learner)
    if cfg.learner.type == 'cbp' and cfg.net.type == 'deep_ffnn_weight_norm_multi_channel_rescale':
        from src.algos.supervised.continuous_backprop_with_GnT import ContinualBackprop_for_FC
        learner = ContinualBackprop_for_FC(net, cfg.learner)


    epochs_per_task = cfg.epochs


    
    # setup evaluation:

    if cfg.use_json:
        # raise not implemented error:
        raise NotImplementedError("JSON logging functionality is not implemented yet.")
        
        # from src.utils.data_logging import save_data_json
        # import json
        # # set up the folder for the json file:
        # # find a exp_id, integer, that is not used in the experiments folder:
        # exp_id = 0
        # dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments',
        #                                   'rank_tracking_in_shifting_task', f'experiment_cfg_{exp_id}')
        
        # run_dir = os.path.join(dir_for_experiment, 'run', f'run_{cfg.run_id}')
        
        # while os.path.exists(dir_for_experiment):
        #     exp_id += 1
        #     dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments', 'basic_training', f'experiment_cfg_{exp_id}')
        #     os.makedirs(dir_for_experiment, exist_ok=True)
        
        
        # # create a file to save the in dir_for_experiment, containing the config
        # os.makedirs(dir_for_experiment, exist_ok=True)
        # with open(os.path.join(dir_for_experiment, 'config.json'), 'w') as f:
        #     json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)




        # # create a folder for the run:
        # os.makedirs(run_dir, exist_ok=True)
    
    if "accuracy" in cfg.evaluation.eval_metrics:
        accuracy = nll_accuracy
    if 'loss' in cfg.evaluation.eval_metrics:
        loss_func  = F.cross_entropy
    # if cfg.evaluation.eval_metrics contains anything else other than 'accuracy', raise not implemented error:
    if any(metric not in ['accuracy', 'loss'] for metric in cfg.evaluation.eval_metrics):
        raise NotImplementedError("Only 'accuracy' and 'loss' evaluation metrics are implemented.")
    
    # wandb setup
    if cfg.use_wandb:
        import wandb
        print('finished importing wandb')
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, config= cfg_dict )
   
    net.train()
    # loop though the tasks
    for task_idx in range(cfg.num_tasks):
        # for each task, set a new permutation:
        # for an FC network, the permutation is a random permutation of the input size
        pixel_permutation = np.random.permutation(cfg.net.netparams.input_size)
        
        #  wrap the dataset with the permutation:
        if cfg.net.network_class == 'fc':
            flatten = True
        if cfg.net.network_class == 'conv':
            flatten = False     
        
        permutated_train_set = PermutedDataset(train_set, permutation=pixel_permutation, flatten=flatten, transform=None)#note: transform was used when setting up the dataset, but not used here.
        
        permutated_train_loader = torch.utils.data.DataLoader(permutated_train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # training loop:

        for epoch in tqdm(range(epochs_per_task), desc='Epoch'):
            
            # running loss for the epoch:
            running_loss = 0.0

            # alternative way to track loss, by batch loss
            batch_running_loss = 0.0
            
            # for accuracy
            number_of_correct = 0
            #number_of_correct_2 =0
            total = 0

            
            for input, label in tqdm(permutated_train_loader, desc=f"Epoch: {epoch}, progress on batches", leave =True):
                if input is None or label is None:
                    print("Found None in the data loader batch")
                    raise ValueError("Found None in the data loader batch")
                
                input, label = input.to(cfg.device), label.to(cfg.device)
                
                # print(label.dtype)  # Should be torch.long for CrossEntropyLoss
                # print(label.min(), label.max())  # Should be within [0, num_classes - 1]
                loss, output = learner.learn(input, label)
                #running_loss+= loss*input.size(0)
                batch_running_loss += loss
                _, predicted = output.max(1)
                total += label.size(0)
                # predicted = predicted.cpu()
                number_of_correct += predicted.eq(label).sum().cpu().item()
                #torch.max(output.data, 1)


            # extrack rank:
            if epoch % cfg.rank_measure_freq_to_epoch == 0 and cfg.track_rank:
                # calculate and log accurary and loss:
                acc = number_of_correct/total
                #acc_2 = number_of_correct_2/total
                #loss = running_loss/total
                loss_by_batch = batch_running_loss/len(permutated_train_loader)
                
                data = {
                    'global_epoch': task_idx*epochs_per_task + epoch,
                    'task_idx': task_idx,
                    'accuracy': acc,
                    #'accuracy_2': acc_2,
                    #'loss': loss,
                    'loss_by_batch': loss_by_batch
                }               
                

                
                # tracking rank:
                if cfg.track_rank:
                    if cfg.track_rank_batch == "last":
                    
                        # get features of the last batch:
                        list_of_features_for_every_layers = learner.previous_features    
                        # a list of dictionaries, each dictionary contains rank proxies for each output layer:
                    if cfg.track_rank_batch == "use_specified":
                        # get features of the specified batch:
                        extracted_list_of_data = [permutated_train_set[i] for i in range(min(cfg.specified_batch_size, len(permutated_train_set)))]
                        extracted_inputs = [item[0] for item in extracted_list_of_data]  # Get the images
                        # Stack the list of tensors into a single batched tensor
                        extracted_inputs = torch.stack(extracted_inputs).to(cfg.device)
                        

                        _, list_of_features_for_every_layers = net.predict(extracted_inputs)
                        
                    # convert all the features into 2d, if they are not already:
                    list_of_features_for_every_layers = [feature.view(feature.size(0), -1) for feature in list_of_features_for_every_layers]
                    
                    
                    if cfg.track_rank_batch == "all":
                        # raise not implemented error:
                        raise NotImplementedError("Tracking rank for all batches is not implemented yet.")
                        

                    rank_summary_list = compute_all_rank_measures_list(
                        features=list_of_features_for_every_layers,
                        use_pytorch_entropy_for_effective_rank=cfg.use_pytorch_entropy_for_effective_rank,
                        prop_for_approx_or_l1_rank=cfg.prop_for_approx_or_l1_rank,
                        numerical_rank_epsilon = cfg.numerical_rank_epsilon)

                # tracking actual rank:
                if cfg.track_actual_rank:
                    actual_rank_list = []
                    for feature in list_of_features_for_every_layers:

                        feature = feature.cpu().detach()
                        feature_actual_rank = torch.linalg.matrix_rank(feature)
                        actual_rank_list.append(feature_actual_rank)
                
                # tracking dead units:
                if cfg.track_dead_units:
                    dead_units_for_features = count_saturated_units_list(
                        features_list=list_of_features_for_every_layers,
                        activation_type=cfg.net.netparams.act_type,
                        threshold=cfg.threshold_for_non_saturating_act)
                    for layer_idx in range(len(list_of_features_for_every_layers)):
                        data[f'layer_{layer_idx}_num_dead_units'] = dead_units_for_features[layer_idx]
                

                
                if cfg.debug_mode:
                    assert len(rank_summary_list) == len(list_of_features_for_every_layers), "The rank summary list and the list of features should have the same length"
                    assert len(actual_rank_list) == len(list_of_features_for_every_layers), "The rank summary list and the list of features should have the same length"
                    
               
                

                
                
                # calculate the effective rank, approximate rank, l1 distribution rank, numerical rank for each layer:
                
                # check!
                for layer_idx in range(len(list_of_features_for_every_layers)):
                    # for each layer,
                    data[f'layer_{layer_idx}_effective_rank'] = rank_summary_list[layer_idx]['effective_rank']
                    data[f'layer_{layer_idx}_approximate_rank'] = rank_summary_list[layer_idx]['approximate_rank']
                    data[f'layer_{layer_idx}_l1_distribution_rank'] = rank_summary_list[layer_idx]['l1_distribution_rank']
                    data[f'layer_{layer_idx}_numerical_rank'] = rank_summary_list[layer_idx]['numerical_rank']
                
                if cfg.track_actual_rank:
                    for layer_idx in range(len(list_of_features_for_every_layers)):
                        data[f'layer_{layer_idx}_actual_rank'] = actual_rank_list[layer_idx]
                
                if cfg.track_weight_magnitude:
                    weight_magnitude_stats = track_weight_stats(net, layer_identifiers=cfg.layers_identifier)
                    # register the weight magnitude stats in the data:
                    for name, stat in weight_magnitude_stats.items():
                        data[name] = stat
                    
                
            # log to wandb:
            if cfg.use_wandb:
                wandb.log(data)
            if cfg.use_json:
                save_data_json(data, run_dir, filename=f'run_{cfg.run_id}.json')

                

        
    

if __name__ == "__main__":
    # clear cached memory:
    torch.cuda.empty_cache()
    
    
    main()
    
    
    

