from configs.configurations import ExperimentConfig
from src.models.model_factory import model_factory
from src.algos.supervised.supervised_factory import create_learner

# Use default experiment config (ConvNet) and verify network_class inference
exp = ExperimentConfig()
exp.net.type = 'ConvNet'  # ensure known conv type
# Provide minimal netparams needed (ConvNet expects input_height/width)
from configs.configurations import NetParams
exp.net.netparams = NetParams(num_classes=10, activation='relu', input_height=28, input_width=28, in_channels=3)
net = model_factory(exp.net)
learner = create_learner(exp.learner, net, exp.net)
print('Inferred learner network_class:', getattr(exp.learner, 'network_class', None))
print('Net type:', exp.net.type)
print('Learner type:', exp.learner.type)
