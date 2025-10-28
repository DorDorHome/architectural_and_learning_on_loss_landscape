"""
Experiment with different training data shift modes for plasticity in neural networks.
Purpose: To analyze how varying the training data shift affects the performance of plasticity mechanisms in neural networks.

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
import random
# import pickle
# import argparse
# import numpy as np
from tqdm import tqdm
import hydra 
import numpy as np #used in concatenating the data
from omegaconf import OmegaConf # , DictConfig
# import algorithm:
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.backprop_with_semantic_features import BackpropWithSemanticFeatures
# import model factory:

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
from src.data_loading.shifting_dataset import Permuted_input_Dataset, Permuted_output_Dataset
# rank tracking:
from src.utils.zeroth_order_features import compute_all_rank_measures_list, count_saturated_units_list
#from src.utils.miscellaneous import compute_matrix_rank_summaries
#from src.utils.partial_jacobian_source import compute_rank_for_list_of_features

# weight magnitude tracking:
from src.utils.track_weights_norm import track_weight_stats

import torchvision.transforms as transforms
import torchvision
import torch

# for checking non-finite loss:
from src.utils.robust_checking_of_training_errors import _log_and_raise_non_finite_error

@hydra.main(config_path="cfg", config_name="rank_track_with_input_vs_output_conv", version_base=None)
def main(cfg: ExperimentConfig) -> Any:
    """
    Main function to run the experiment with different training data shift modes.
    """
    # Ensure reproducibility by setting the seed
    if cfg.seed is None or not isinstance(cfg.seed, (int, float)):
        cfg.seed = random.randint(0, 2**32 - 1)  # Generate a random seed if not provided

    # Log the seed for reproducibility tracking
    print(f"Using seed: {cfg.seed}")

    # Set seed for Python, NumPy, and PyTorch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)  # If using multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility

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
        # Use enhanced learner that provides semantic layer names
        learner = BackpropWithSemanticFeatures(net, cfg.learner, cfg.net)
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
        
        from src.utils.data_logging import save_data_json
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
        try:
            from src.utils.task_shift_logging import build_logging_config_dict
            cfg_dict = build_logging_config_dict(cfg)
        except Exception as e:
            print(f"Warning: task shift logging sanitization failed, using full config. Error: {e}")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, config= cfg_dict )
   
    net.train()
    
    # Wrap the entire training in try-catch to handle NaN properly
    try:
        # loop though the tasks
        for task_idx in range(cfg.num_tasks):
            print(f"\n{'='*50}")
            print(f"Starting task {task_idx+1}/{cfg.num_tasks}")
            print(f"{'='*50}")
            
            if cfg.task_shift_mode == 'input_permutation':
                # for each task, set a new permutation for pixels
                # for an FC network, the permutation is a random permutation of the input size
                pixel_permutation = np.random.permutation(cfg.net.netparams.input_height * cfg.net.netparams.input_width)
                
                #  wrap the dataset with the permutation:
                if cfg.net.network_class == 'fc':
                    flatten = True
                if cfg.net.network_class == 'conv':
                    flatten = False 
                
                permutated_train_set = Permuted_input_Dataset(train_set, permutation=pixel_permutation, flatten=flatten, transform=None)#note: transform was used when setting up the dataset, but not used here.
                
                permutated_train_loader = torch.utils.data.DataLoader(permutated_train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            elif cfg.task_shift_mode == 'output_permutation':
                # for each task, set a new permutation for the output classes
                # for an FC network, the permutation is a random permutation of the number of classes
                class_permutation = np.random.permutation(cfg.data.num_classes)
                
                # wrap the dataset with the permutation:
                permutated_train_set = Permuted_output_Dataset(train_set, permutation=class_permutation)
                
                permutated_train_loader = torch.utils.data.DataLoader(permutated_train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
                
        
        
        # training loop:

            for epoch in range(epochs_per_task):
                print(f"Epoch {epoch+1}/{epochs_per_task}")
                
                # running loss for the epoch:
                running_loss = 0.0

                # alternative way to track loss, by batch loss
                batch_running_loss = 0.0
                
                # for accuracy
                number_of_correct = 0
                #number_of_correct_2 =0
                total = 0

                
                for batch_idx, (input, label) in enumerate(tqdm(permutated_train_loader, desc=f"Epoch: {epoch}, progress on batches", leave =True)):
                    if input is None or label is None:
                        print("Found None in the data loader batch")
                        raise ValueError("Found None in the data loader batch")
                    
                    input, label = input.to(cfg.device), label.to(cfg.device)
                    
                    # print(label.dtype)  # Should be torch.long for CrossEntropyLoss
                    # print(label.min(), label.max())  # Should be within [0, num_classes - 1]
                    loss, output = learner.learn(input, label)
                    
                    # Critical: Check for non-finite loss and STOP immediately
                    if not torch.isfinite(torch.as_tensor(loss)):
                        _log_and_raise_non_finite_error(task_idx, epoch,
                                        batch_idx, loss,
                                        input, label,
                                        output, learner, net)
                        # This should never be reached due to the exception above
                        raise RuntimeError("Training should have stopped due to NaN loss!")

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
                        
                        # Enhanced logging with semantic layer names
                        # This demonstrates the new capability while keeping existing code working
                        try:
                            if hasattr(learner, 'get_layer_names'):
                                layer_names = learner.get_layer_names()
                                # Log rank metrics with semantic names (in addition to indexed names)
                                for i, layer_name in enumerate(layer_names):
                                    if i < len(rank_summary_list):
                                        # Add semantic name versions of the metrics
                                        data[f'{layer_name}_effective_rank'] = rank_summary_list[i]['effective_rank']
                                        data[f'{layer_name}_approximate_rank'] = rank_summary_list[i]['approximate_rank']
                                        data[f'{layer_name}_l1_distribution_rank'] = rank_summary_list[i]['l1_distribution_rank']
                                        data[f'{layer_name}_numerical_rank'] = rank_summary_list[i]['numerical_rank']
                            else:
                                # Learner doesn't have get_layer_names(), use indexed names
                                for layer_idx in range(len(list_of_features_for_every_layers)):
                                    # for each layer,
                                    data[f'layer_{layer_idx}_effective_rank'] = rank_summary_list[layer_idx]['effective_rank']
                                    data[f'layer_{layer_idx}_approximate_rank'] = rank_summary_list[layer_idx]['approximate_rank']
                                    data[f'layer_{layer_idx}_l1_distribution_rank'] = rank_summary_list[layer_idx]['l1_distribution_rank']
                                    data[f'layer_{layer_idx}_numerical_rank'] = rank_summary_list[layer_idx]['numerical_rank']
                        except Exception as e:
                            # Fallback: if semantic names fail, use indexed names
                            print(f"Semantic layer naming failed, using indexed names as fallback: {e}")
                            for layer_idx in range(len(list_of_features_for_every_layers)):
                                # for each layer,
                                data[f'layer_{layer_idx}_effective_rank'] = rank_summary_list[layer_idx]['effective_rank']
                                data[f'layer_{layer_idx}_approximate_rank'] = rank_summary_list[layer_idx]['approximate_rank']
                                data[f'layer_{layer_idx}_l1_distribution_rank'] = rank_summary_list[layer_idx]['l1_distribution_rank']
                                data[f'layer_{layer_idx}_numerical_rank'] = rank_summary_list[layer_idx]['numerical_rank']

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
                            activation_type=cfg.net.netparams.activation,
                            threshold=cfg.threshold_for_non_saturating_act)
                        for layer_idx in range(len(list_of_features_for_every_layers)):
                            data[f'layer_{layer_idx}_num_dead_units'] = dead_units_for_features[layer_idx]
                    

                    
                    if cfg.debug_mode:
                        assert len(rank_summary_list) == len(list_of_features_for_every_layers), "The rank summary list and the list of features should have the same length"
                        assert len(actual_rank_list) == len(list_of_features_for_every_layers), "The rank summary list and the list of features should have the same length"
                    
                    if cfg.track_actual_rank:
                        for layer_idx in range(len(list_of_features_for_every_layers)):
                            data[f'layer_{layer_idx}_actual_rank'] = actual_rank_list[layer_idx]
                    
                    if cfg.track_weight_magnitude:
                        weight_magnitude_stats = track_weight_stats(net, layer_identifiers=cfg.layers_identifier)
                        # register the weight magnitude stats in the data with clear naming:
                        for name, stat in weight_magnitude_stats.items():
                            # Ensure the key clearly indicates this is mean absolute weight value
                            if not name.endswith('_mean_abs_weight'):
                                data[f'{name}_mean_abs_weight'] = stat
                            else:
                                data[name] = stat
                        
                    
                    # log to wandb:
                    if cfg.use_wandb:
                        wandb.log(data)
                    if cfg.use_json:
                        save_data_json(data, run_dir, filename=f'run_{cfg.run_id}.json')

    except ValueError as e:
        # NaN/Inf errors will propagate as ValueError - let them stop the script
        print(f"Training stopped due to numerical instability: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error during training: {e}")
        raise

if __name__ == "__main__":
    # clear cached memory:
    torch.cuda.empty_cache()
    
    
    main()




