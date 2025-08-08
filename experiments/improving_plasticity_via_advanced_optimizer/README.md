# Experiment: Improving Plasticity via Advanced Optimizer

This experiment investigates the effects of using an advanced optimizer, specifically one that incorporates a "generate and test" mechanism, on the plasticity of a neural network. The goal is to see if this type of optimizer can help the network adapt to changing data distributions more effectively than a standard optimizer.

## Description

The experiment uses a similar setup to the `output_vs_input_plasticity` experiment. A neural network is trained on a task where the data distribution shifts over time. The key difference is the use of the `basic_continous_backprop` learner, which employs a "generate and test" strategy to replace neurons that are no longer useful.

The experiment tracks various metrics, including:
- Loss and accuracy
- Rank of the feature representations at different layers
- Weight magnitudes

These metrics are logged to `wandb` for analysis.

## How to Run

The experiment is configured using `hydra`. To run the experiment, you can use the following command from the root of the repository:

```bash
python experiments/improving_plasticity_via_advanced_optimizer/train_with_improved_optimizer.py
```

You can override the configuration parameters from the command line. For example, to run the experiment with a different learning rate, you can use:

```bash
python experiments/improving_plasticity_via_advanced_optimizer/train_with_improved_optimizer.py learner.step_size=0.001
```

To disable `wandb` logging for local testing, you can use:

```bash
python experiments/improving_plasticity_via_advanced_optimizer/train_with_improved_optimizer.py use_wandb=False
```
