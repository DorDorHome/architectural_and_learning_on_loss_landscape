from typing import Any
import sys
import pathlib
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

# import the single_run code:
from experiments.basic_training.single_run import main as single_run_main



@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg):
    
    if cfg.use_json:
        from src.utils.data_logging import save_data_json
        import json
        # set up the folder for the json file:
        # find a exp_id, integer, that is not used in the experiments folder:
        exp_id = 0
        dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments', 'basic_training', f'experiment_cfg_{exp_id}')
        while os.path.exists(dir_for_experiment):
            exp_id += 1
            dir_for_experiment = os.path.join(PROJECT_ROOT, 'experiments', 'basic_training', f'experiment_cfg_{exp_id}')
            os.makedirs(dir_for_experiment, exist_ok=True)
        
    for run_id in range(cfg.runs):
        print(f"Run {run_id}")
        # set config for each run, with unique run_id
        cfg_single_run = cfg
        cfg_single_run.run_id = run_id
        cfg_single_run.runs = 1
        single_run_main(cfg_single_run)
        
if __name__ == "__main__":
    main()
    