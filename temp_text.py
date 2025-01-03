# tests/test_config/test_config.py

import pytest
from hydra import initialize, compose
from configs.configurations import ExperimentConfig  # 'configs' is now a recognized package

def test_config_loading():
    # Initialize Hydra with the path to the configurations directory
    with initialize(config_path="../../configs", job_name="test_job"):
        cfg = compose(config_name="experiment_config")

    # Convert Hydra's DictConfig to the ExperimentConfig dataclass
    experiment_cfg = ExperimentConfig(**cfg.experiments)
    
    # Perform assertions to verify the correctness
    assert experiment_cfg.runs == 1
    assert experiment_cfg.seed == 42
    assert experiment_cfg.device == 'cuda'
    assert experiment_cfg.epochs == 10
    assert experiment_cfg.data.dataset == 'mnist'
    assert experiment_cfg.data.num_classes == 10
    assert experiment_cfg.net.type == 'ConvNet'
    assert experiment_cfg.net.num_classes == 10
    assert experiment_cfg.learner.type == 'backprop'
    assert experiment_cfg.learner.opt == 'adam'
    assert experiment_cfg.learner.loss == 'nll'
    assert experiment_cfg.evaluation["eval_freq_epoch"] == 1
    assert experiment_cfg.evaluation["eval_metrics"] == ['accuracy', 'loss']
    assert experiment_cfg.evaluation["save_dir"] == 'results_raw'
    assert experiment_cfg.evaluation["save_name"] == 'basic_training'

def test_config_constraints():
    # Initialize Hydra with the path to the configurations directory
    with initialize(config_path="../../configs", job_name="test_job"):
        # Override num_classes to create a mismatch
        cfg = compose(
            config_name="experiment_config",
            overrides=[
                "experiments.net.num_classes=5",
                "experiments.data.num_classes=10"
            ]
        )

    # Attempt to create the ExperimentConfig and expect a ValueError
    with pytest.raises(ValueError, match="net.num_classes must match data.num_classes"):
        ExperimentConfig(**cfg.experiments)