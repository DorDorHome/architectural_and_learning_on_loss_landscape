# for testing purpose only can be safely deleted.

from configs.configurations import ExperimentConfig
exp = ExperimentConfig()
print('task_shift_mode =', exp.task_shift_mode)
print('task_shift_param =', exp.task_shift_param)
print('num_tasks =', exp.num_tasks)
