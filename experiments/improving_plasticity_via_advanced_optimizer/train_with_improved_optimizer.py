"""
Experiment with different training data shift modes for plasticity in neural networks.
Purpose: To analyze how varying the training data shift affects the performance of plasticity mechanisms in neural networks.

"""



import warnings
warnings.filterwarnings("ignore", message="A NumPy version")
warnings.filterwarnings("ignore", message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


from typing import Any
import sys
import pathlib
import torch.utils

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.configurations import ExperimentConfig
import json
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
# from src.algos.supervised.basic_backprop import Backprop
from src.algos.supervised.supervised_factory import create_learner
# import model factory:

from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
import torch.nn.functional as F
from src.utils.miscellaneous import nll_accuracy
from src.data_loading.shifting_dataset import (
    create_stateful_dataset_wrapper,
    create_stateless_dataset_wrapper
)
# rank tracking:
from src.utils.zeroth_order_features import compute_all_rank_measures_list, count_saturated_units_list
# rank drop dynamics:
from src.utils.rank_drop_dynamics import compute_rank_dynamics_from_features
#from src.utils.miscellaneous import compute_matrix_rank_summaries
#from src.utils.partial_jacobian_source import compute_rank_for_list_of_features

# weight magnitude tracking:
from src.utils.track_weights_norm import track_weight_stats

import torchvision.transforms as transforms
import torchvision
import torch
import torch.multiprocessing as mp

# for checking non-finite loss:
from src.utils.robust_checking_of_training_errors import _log_and_raise_non_finite_error




@hydra.main(config_path="cfg", config_name="config", version_base=None)
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

    # Apply cuda:1 workarounds if enabled in config
    if hasattr(cfg, 'enable_cuda1_workarounds') and cfg.enable_cuda1_workarounds:
        os.environ['SIGMA_FORCE_CPU_EIGH'] = '1'
        os.environ['LLA_PREFER_GPU_EIGH'] = '0'
        print("⚠️  cuda:1 workarounds enabled - using CPU eigendecomposition")
        print("   (SIGMA_FORCE_CPU_EIGH=1, LLA_PREFER_GPU_EIGH=0)")
        print("   Set enable_cuda1_workarounds=False in config to disable")



    print(OmegaConf.to_yaml(cfg))
    # set up the transform being used for the dataset, given the model architecture and dataset
    # expected cfg.data.dataset value is 'MNIST'
    # expected cfg.net.type value is 'ConvNet'
    transform = transform_factory(cfg.data.dataset, cfg.net.type)

    # setup data:
    train_set, _ = dataset_factory(cfg.data, transform = transform, with_testset= False)            # The number of workers should be set from the config, not hardcoded.
    # This was a temporary fix for a pickling issue that is now resolved.
    if cfg.num_workers == "auto":
        # Set num_workers to the number of available CPU cores
        num_workers = int(os.cpu_count() or 0)
        print(f"Dynamically setting num_workers to {num_workers}")
    else:
        num_workers = cfg.num_workers
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
    learner = create_learner(cfg.learner, net, cfg.net)


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
            print(f"Warning: task shift logging sanitization failed, falling back to full config. Error: {e}")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project=cfg.wandb.project, config= cfg_dict )

    net.train()


    # Determine if the shift mode is stateful. This covers cases in which the wrapper changes across tasks
    is_stateful = cfg.task_shift_mode in ["drifting_values", 'continuous_input_deformation']
    # Initialize the appropriate wrapper if it's stateful
    dataset_wrapper = None
    if is_stateful:
        dataset_wrapper = create_stateful_dataset_wrapper(cfg, train_set)

    # Configure DataLoader workers (safe defaults) and seeding
    try:
        num_workers = int(os.cpu_count() or 0) if str(cfg.num_workers).lower() == "auto" else int(cfg.num_workers)
    except Exception:
        num_workers = 0
    # Stateful wrappers are often not multiprocess-safe; prefer single worker to surface errors clearly
    if is_stateful and num_workers != 0:
        print("Info: Forcing num_workers=0 for stateful dataset wrapper to avoid worker pickling/multiprocessing issues.")
        num_workers = 0

    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Wrap the entire training in try-catch to handle NaN properly
    try:
        # loop though the tasks
        for task_idx in range(cfg.num_tasks):
            print(f"\n{'='*50}")
            print(f"Starting task {task_idx+1}/{cfg.num_tasks}")
            print(f"{'='*50}")

            # --- Dataset Handling for each task ---
            if is_stateful:
                # For stateful shifts, update the wrapper's state for the new task
                if hasattr(dataset_wrapper, 'update_task'):
                    dataset_wrapper.update_task()
                current_train_set = dataset_wrapper if dataset_wrapper is not None else train_set
            else:
                # For stateless shifts, use the factory to get a new wrapper for each task
                current_train_set = create_stateless_dataset_wrapper(cfg, train_set, task_idx) or train_set

            train_loader = torch.utils.data.DataLoader(
                current_train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                worker_init_fn=_seed_worker if num_workers > 0 else None
            )

            # Update drift for stateful datasets that require it
            if is_stateful and hasattr(dataset_wrapper, 'update_drift'):
                print("Updating dataset drift...")
                dataset_wrapper.update_drift()

            pbar = tqdm(range(epochs_per_task), desc=f'Task {task_idx+1}/{cfg.num_tasks}')
            for epoch in pbar:
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0


                for batch_idx, (input, target) in enumerate(train_loader):
                    input = input.to(cfg.net.device)
                    if cfg.task_shift_mode == 'drifting_values':
                        drifting_values, original_labels = target
                        input = input.to(cfg.net.device)
                        drifting_values = drifting_values.to(cfg.net.device)
                        original_labels = original_labels.to(cfg.net.device)

                        if cfg.debug_mode:
                            # Debug: Check for problematic values
                            if batch_idx == 0:  # Only for first batch to avoid spam
                                print(f"Debug: Drifting values - min: {drifting_values.min():.3f}, max: {drifting_values.max():.3f}, "
                                    f"mean: {drifting_values.mean():.3f}, std: {drifting_values.std():.3f}")
                                print(f"Debug: Has NaN in drifting_values: {torch.isnan(drifting_values).any()}")
                                print(f"Debug: Has Inf in drifting_values: {torch.isinf(drifting_values).any()}")
                                print(f"Debug: Original labels range: {original_labels.min()}-{original_labels.max()}")

                        loss, output = learner.learn_from_partial_values(input, drifting_values, original_labels)

                        if cfg.debug_mode:
                            # Debug: Check outputs and loss
                            if batch_idx == 0:
                                print(f"Debug: Network output - min: {output.min():.3f}, max: {output.max():.3f}, "
                                    f"mean: {output.mean():.3f}, std: {output.std():.3f}")
                                print(f"Debug: Has NaN in output: {torch.isnan(output).any()}")
                                print(f"Debug: Loss: {loss:.6f}, isnan: {torch.isnan(torch.tensor(loss))}")

                    else:
                        input = input.to(cfg.net.device)
                        target = target.to(cfg.net.device)
                        loss, output = learner.learn(input, target)

                    # Check for non-finite loss and handle it
                    # _log_and_raise_non_finite_error( loss, epoch, batch_idx, task_idx, cfg)



                    # Accuracy calculation is not applicable for the regression task in 'drifting_values' mode.
                    if cfg.task_shift_mode != 'drifting_values':
                        with torch.no_grad():
                            _, predicted = torch.max(output, dim=1)
                            correct = predicted.eq(target).sum().item()
                            epoch_correct += correct

                    epoch_total += input.size(0)
                    epoch_loss += loss.item() if torch.is_tensor(loss) else loss

                # extrack rank:
                if epoch % cfg.rank_measure_freq_to_epoch == 0 and cfg.track_rank:
                    # calculate and log accurary and loss:
                    # acc = number_of_correct/total
                    #acc_2 = number_of_correct_2/total
                    #loss = running_loss/total
                    # loss_by_batch = batch_running_loss/len(current_train_loader)

                    data = {
                        'global_epoch': task_idx*epochs_per_task + epoch,
                        'task_idx': task_idx,
                        #'accuracy': acc,
                        #'accuracy_2': acc_2,
                        # "loss_by_batch": loss,
                        'epoch_loss': epoch_loss / len(train_loader),
                    }
                    if cfg.task_shift_mode != 'drifting_values':
                            data['epoch_accuracy'] = epoch_correct / epoch_total



                    # tracking rank:
                    if cfg.track_rank:
                        if cfg.track_rank_batch == "last":

                            # get features of the last batch:
                            list_of_features_for_every_layers = learner.previous_features
                            # a list of dictionaries, each dictionary contains rank proxies for each output layer:
                        if cfg.track_rank_batch == "use_specified":
                            # get features of the specified batch:
                            extracted_list_of_data = [current_train_set[i] for i in range(min(cfg.specified_batch_size, len(current_train_set)))]
                            extracted_inputs = [item[0] for item in extracted_list_of_data]  # Get the images
                            # Stack the list of tensors into a single batched tensor
                            extracted_inputs = torch.stack(extracted_inputs).to(cfg.net.device)


                            _, list_of_features_for_every_layers = net.predict(extracted_inputs)

                        # convert all the features into 2d, if they are not already:
                        list_of_features_for_every_layers = [feature.view(feature.size(0), -1) for feature in list_of_features_for_every_layers]

                        # Debug: Print feature shapes and statistics before rank computation
                        if cfg.debug_mode:
                            print("Debug: Features for rank computation:")
                            print(f"Debug: Processing {len(list_of_features_for_every_layers)} feature layers")
                            for i, feature in enumerate(list_of_features_for_every_layers):
                                print(f"Layer {i}: shape={feature.shape}, min={feature.min():.2e}, max={feature.max():.2e}, "
                                    f"std={feature.std():.2e}, norm={torch.linalg.norm(feature):.2e}")


                        if cfg.track_rank_batch == "all":
                            # raise not implemented error:
                            raise NotImplementedError("Tracking rank for all batches is not implemented yet.")


                        rank_summary_list = compute_all_rank_measures_list(
                            features=list_of_features_for_every_layers,
                            use_pytorch_entropy_for_effective_rank=cfg.use_pytorch_entropy_for_effective_rank,
                            prop_for_approx_or_l1_rank=cfg.prop_for_approx_or_l1_rank,
                            numerical_rank_epsilon = cfg.numerical_rank_epsilon)

                        # Compute rank drop dynamics if enabled
                        if 'track_rank_drop' in cfg and cfg.track_rank_drop:
                            try:
                                rank_dynamics = compute_rank_dynamics_from_features(
                                    feature_list=list_of_features_for_every_layers,
                                    rank_summary_list=rank_summary_list,
                                    batch_size=cfg.specified_batch_size if cfg.track_rank_batch == "use_specified" else cfg.batch_size,
                                    mode=cfg.rank_drop_mode,  # Use ratio mode as default
                                    use_theoretical_max_first=cfg.from_theoretical_max_first_feature_rank
                                )

                                # Add rank dynamics to data dict
                                for metric_name, value in rank_dynamics.items():
                                    data[metric_name] = value

                                if cfg.debug_mode:
                                    print(f"Debug: Computed rank dynamics metrics: {list(rank_dynamics.keys())}")
                                    for metric_name, value in rank_dynamics.items():
                                        print(f"  {metric_name}: {value:.4f}")

                            except Exception as e:
                                print(f"Warning: Failed to compute rank drop dynamics: {e}")
                                if cfg.debug_mode:
                                    import traceback
                                    print(f"Full traceback: {traceback.format_exc()}")

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
                        if cfg.track_actual_rank:
                            assert len(actual_rank_list) == len(list_of_features_for_every_layers), "The actual rank list and the list of features should have the same length"

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


                    # Log drifting values for 'drifting_values' mode
                    if cfg.task_shift_mode == 'drifting_values' and hasattr(current_train_set, 'values'):
                        for label_idx in range(len(current_train_set.values)):
                            data[f'label_{label_idx}_value'] = current_train_set.values[label_idx].item()

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

    # Set the start method for multiprocessing to 'spawn' to avoid CUDA errors.
    # This is necessary when using CUDA with num_workers > 0 in DataLoader.
    try:
        mp.set_start_method('spawn', force=True)
        print("multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass  # context has already been set.

    # clear cached memory:
    torch.cuda.empty_cache()


    main()
