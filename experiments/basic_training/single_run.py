
"""

The most basic experiment, to replicate a simple training loop.
Only use this is runs is one.

"""

from typing import Any
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from configs.configurations import (
    ExperimentConfig,
    BackpropConfig,
    ContinuousBackpropConfig,
    RRContinuousBackpropConfig,
)
from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.rr_cbp_fc import RankRestoringCBP_for_FC
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
from src.models.model_factory import model_factory
from src.utils.miscellaneous import nll_accuracy
from src.utils.zeroth_order_features import compute_all_rank_measures_list


def compute_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)
        correct = predicted.eq(target).sum().item()
        total = target.size(0)
        return correct / total


@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg: ExperimentConfig):
    print(OmegaConf.to_yaml(cfg))

    net = model_factory(cfg.net)

    if getattr(net, 'type', None) is None:
        inferred_cls = getattr(cfg.net, 'network_class', None)
        if inferred_cls is None and hasattr(cfg.net, 'type'):
            net_type_str = str(cfg.net.type).lower()
            if any(token in net_type_str for token in ['ffnn', 'mlp', 'fc']):
                inferred_cls = 'fc'
            elif any(token in net_type_str for token in ['conv', 'resnet', 'vgg']):
                inferred_cls = 'conv'
        if inferred_cls == 'fc':
            setattr(net, 'type', 'FC')
        elif inferred_cls == 'conv':
            setattr(net, 'type', 'ConvNet')

    if cfg.runs != 1:
        raise ValueError("This script is only for single runs. Please set runs to 1 or use another script.")

    if cfg.run_id is None:
        cfg.run_id = 0

    learner_cfg = cfg.learner

    learner_dict = None
    if isinstance(learner_cfg, DictConfig):
        learner_dict = OmegaConf.to_container(learner_cfg, resolve=True)
        learner_type = learner_dict.get('type', '').lower()
    else:
        learner_type = getattr(learner_cfg, 'type', '').lower()

    if learner_type == 'backprop':
        if not isinstance(cfg.learner, BackpropConfig):
            if learner_dict is None:
                learner_dict = OmegaConf.to_container(cfg.learner, resolve=True)
            cfg.learner = BackpropConfig(**learner_dict)
        learner = Backprop(net, cfg.learner)
    elif learner_type in {'cbp', 'continuous_backprop'}:
        from src.algos.supervised.continuous_backprop_with_GnT import (
            ContinualBackprop_for_FC,
            ContinuousBackprop_for_ConvNet,
        )

        if getattr(cfg.net, 'network_class', None) == 'conv' or str(cfg.net.type).lower() in {'conv_net', 'convnet'}:
            if not isinstance(cfg.learner, ContinuousBackpropConfig):
                if learner_dict is None:
                    learner_dict = OmegaConf.to_container(cfg.learner, resolve=True)
                cfg.learner = ContinuousBackpropConfig(**learner_dict)
            learner = ContinuousBackprop_for_ConvNet(net, cfg.learner)
        else:
            if not isinstance(cfg.learner, ContinuousBackpropConfig):
                if learner_dict is None:
                    learner_dict = OmegaConf.to_container(cfg.learner, resolve=True)
                cfg.learner = ContinuousBackpropConfig(**learner_dict)
            learner = ContinualBackprop_for_FC(net, cfg.learner)
    elif learner_type == 'rr_cbp':
        if not isinstance(cfg.learner, RRContinuousBackpropConfig):
            if learner_dict is None:
                learner_dict = OmegaConf.to_container(cfg.learner, resolve=True)
            cfg.learner = RRContinuousBackpropConfig(**learner_dict)
        if getattr(net, 'type', None) != 'FC':
            setattr(net, 'type', 'FC')
        learner = RankRestoringCBP_for_FC(net, cfg.learner, cfg.net)
    else:
        raise ValueError(f"Unsupported learner type: {cfg.learner.type}")
    
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

    # setup evaluation:
    # wandb setup
    if cfg.use_wandb:
        import wandb
        print('finished importing wandb')
        try:
            from src.utils.task_shift_logging import build_logging_config_dict
            cfg_dict = build_logging_config_dict(cfg)
        except Exception as e:
            print(f"Warning: task shift logging sanitization failed, falling back to full config. Error: {e}")
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

            # ---------------------------------------------------------
            # Step 8: Optional rank tracking of intermediate features.
            # We reuse the last batch seen (input, output) since learner.predict not stored.
            # For more rigorous evaluation use a held-out eval loader later.
            # ---------------------------------------------------------
            if getattr(cfg, 'track_rank', False):
                try:
                    # Run a forward pass (no grad) to collect features via model.predict
                    net.eval()
                    with torch.no_grad():
                        # Use a small subset: reuse final batch 'input' still in scope
                        y_pred, feature_list = net.predict(input)
                        # Ensure 2D (batch, feature_dim) for each feature; flatten spatial dims if needed
                        processed = []
                        for f in feature_list:
                            if f.dim() > 2:
                                processed.append(f.view(f.size(0), -1))
                            else:
                                processed.append(f)
                        rank_measures = compute_all_rank_measures_list(
                            processed,
                            use_pytorch_entropy_for_effective_rank=getattr(cfg, 'use_pytorch_entropy_for_effective_rank', True),
                            prop_for_approx_or_l1_rank=getattr(cfg, 'prop_for_approx_or_l1_rank', 0.99),
                            numerical_rank_epsilon=getattr(cfg, 'numerical_rank_epsilon', 1e-2)
                        )
                        # Flatten rank dicts into logging schema
                        for idx, rm in enumerate(rank_measures):
                            for k, v in rm.items():
                                key = f"rank_layer{idx}_{k}"
                                # Convert tensors to scalars
                                data[key] = float(v.detach().cpu().item())
                    net.train()
                except Exception as e:
                    print(f"Rank tracking failed at epoch {epoch}: {e}")
            # log to wandb:
            if cfg.use_wandb:
                wandb.log(data)
            if cfg.use_json:
                save_data_json(data, run_dir, filename=f'run_{cfg.run_id}.json')

                

        
    

if __name__ == "__main__":
    # clear cached memory:
    torch.cuda.empty_cache()
    
    
    main()
    
    