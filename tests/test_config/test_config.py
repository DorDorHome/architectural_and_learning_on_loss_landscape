
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

    # convert hydra dictconfig to dataclass:
    experiment_cfg = ExperimentConfig(**cfg.experiments)
    
    

    assert is_dataclass(experiment_cfg)
    
    assert isinstance(experiment_cfg, ExperimentConfig)

    # Assertions to verify the correctness
    assert experiment_cfg.runs == 1
    assert experiment_cfg.seed == 42
    assert experiment_cfg.device == 'cuda'
    assert experiment_cfg.epochs == 10  # Assuming 'epochs' is added to ExperimentConfig
    assert experiment_cfg.data.dataset == 'mnist'
    assert experiment_cfg.data.num_classes == 10
    assert experiment_cfg.net.type == 'ConvNet'
    assert experiment_cfg.net.num_classes == 10
    assert experiment_cfg.learner.type == 'backprop'
    assert experiment_cfg.learner.opt == 'adam'
    assert experiment_cfg.learner.loss == 'cross_entropy'
    assert experiment_cfg.evaluation.eval_freq_epoch == 1
    assert experiment_cfg.evaluation.eval_metrics == ['accuracy', 'loss']
    assert experiment_cfg.evaluation.save_dir == 'results_raw'
    assert experiment_cfg.evaluation.save_name == 'basic_training'

# test configuration constraints

def test_config_constraints():
    # Create a configuration with mismatched num_classes
    with initialize(config_path="../../experiments/basic_training/cfg",version_base="1.1"):
        # Manually compose a config with mismatched num_classes
        cfg = compose(
            overrides=[
                "experiments.net.num_classes=5",
                "experiments.data.num_classes=10"
            ],
            config_name="basic_config"
        )
    
    # Attempt to create the ExperimentConfig and expect a ValueError
    with pytest.raises(ValueError, match="net.num_classes must match data.num_classes"):
        ExperimentConfig(**cfg.experiments)