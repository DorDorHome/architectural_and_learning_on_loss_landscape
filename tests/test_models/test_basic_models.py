import pytest

# from dataclasses import is_dataclass

# for 
from hydra import compose, initialize

import sys
from pathlib import Path
from tests.conftest import PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.conv_net import ConvNet