
import pytest
"""
This module contains tests for the configuration of the project.

Modules:
    pytest: A framework that makes building simple and scalable test cases easy.
    dataclasses.is_dataclass: A utility to check if an object is a dataclass.
    hydra.compose: A function to compose a configuration from multiple sources.
    hydra.initialize: A function to initialize the Hydra environment.
    hydra.core.config_store.ConfigStore: A class to register and manage configuration objects.
Constants:
    PROJECT_ROOT: The root directory of the project, resolved to the parent of the current file's parent directory.
Imports:
    ExperimentConfig: The configuration class for the experiment.
    cs: The configuration store instance.

Functions:
    # setup configuration store: Placeholder for setting up the configuration store.
    # test configuration loading: Placeholder for testing the loading of configurations.
    # test configuration constraints: Placeholder for testing the constraints of configurations.
"""
from dataclasses import is_dataclass
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

# locate project root directory
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.configurations import ExperimentConfig, cs

# setup configuration store


# test configuration loading
def test_config_loading():
    """Test the loading of configurations."""
    with initialize(config_path='../../experiments/basic_training/cfg', version_base="1.1"):
        cfg = compose(config_name="basic_config")

    # The config is now loaded at the top level
    experiment_cfg = ExperimentConfig(**cfg)
    
    assert is_dataclass(experiment_cfg)
    assert isinstance(experiment_cfg, ExperimentConfig)

    # Assertions to verify the types
    assert isinstance(experiment_cfg.runs, int)
    assert isinstance(experiment_cfg.seed, int)
    assert isinstance(experiment_cfg.device, str)
    assert isinstance(experiment_cfg.epochs, int)
    assert isinstance(experiment_cfg.data.dataset, str)
    assert isinstance(experiment_cfg.data.num_classes, int)
    assert isinstance(experiment_cfg.net.type, str)
    assert isinstance(experiment_cfg.learner.type, str)
    assert isinstance(experiment_cfg.learner.opt, str)
    assert isinstance(experiment_cfg.learner.loss, str)
    assert isinstance(experiment_cfg.evaluation.eval_freq_epoch, int)
    
    from omegaconf import ListConfig
    assert isinstance(experiment_cfg.evaluation.eval_metrics, (list, ListConfig))

# test configuration constraints

def test_config_constraints():
    # This test is disabled because the logic it was testing has been removed.
    pass
