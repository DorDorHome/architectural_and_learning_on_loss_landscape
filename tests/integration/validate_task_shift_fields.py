import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# for testing purpose only can be safely deleted.

from configs.configurations import ExperimentConfig
exp = ExperimentConfig()
print('task_shift_mode =', exp.task_shift_mode)
print('task_shift_param =', exp.task_shift_param)
print('num_tasks =', exp.num_tasks)
