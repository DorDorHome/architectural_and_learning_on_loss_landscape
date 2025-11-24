import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.configurations import ExperimentConfig, NetParams, ContinuousBackpropConfig
from src.models.model_factory import model_factory
from src.algos.supervised.supervised_factory import create_learner

aliases = ['cbp', 'continuous_backprop', 'basic_continous_backprop']
results = {}
for alias in aliases:
    exp = ExperimentConfig()
    exp.net.type = 'ConvNet'
    exp.net.netparams = NetParams(num_classes=10, activation='relu', input_height=28, input_width=28, in_channels=3)
    # Replace learner with ContinuousBackpropConfig for cbp aliases
    exp.learner = ContinuousBackpropConfig(type=alias, device=exp.device)
    exp.learner.network_class = None  # force inference
    try:
        net = model_factory(exp.net)
        learner = create_learner(exp.learner, net, exp.net)
        results[alias] = f"OK (normalized -> {learner.__class__.__name__}, inferred={exp.learner.network_class})"
    except Exception as e:
        results[alias] = f"FAIL: {e}";

for k,v in results.items():
    print(k, '=>', v)
