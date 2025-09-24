from configs.configurations import EvaluationConfig, ExperimentConfig

e = EvaluationConfig()
print("EvaluationConfig.use_testset=", e.use_testset)
exp = ExperimentConfig()
print("ExperimentConfig.evaluation.use_testset=", exp.evaluation.use_testset)
