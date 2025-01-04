import importlib# for dynamic import
# for support of all models available in torchvision:
from torchvision import models as torchvision_models
from typing import Any
# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from conv_net import ConvNet
from VGG_scratch import VGG


from configs.configurations import NetConfig  # Assuming NetParams is defined here

