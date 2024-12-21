
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

from src.utils.miscellaneous import nll_accuracy as accuracy



@hydra.main(config_path="cfg", config_name="basic_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    # initialize Hydra:

if __name__ == "__main__":
    main()
    
    