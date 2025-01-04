# architectural_and_learning_on_loss_landscape

# to-do:

- test_config.py (done!)


write model loading.


- write backprop
- test_backprop.py 
- generate results_raw from basic_config
- plot results


For each subfolder in experiments (test on basic_training):
- design of experiments (many runs) vs single run
- build generate_cfg_for_different_hyperparam.py
- this should generate many cfg corresponding to different hyperparm for experiments
- design save logic for single_expr, so that different settings can be saved in results/results_raw 
- 


- use superivsed_factory.py to centralize the creation of learners in src.algos.supervised
- test the function

-- design the ideal data format. Does loading the line-by-line as in loss-of-plasticity offer any advantages? 




- build 


basic folder structure:

project/
├── src/
│   ├── algos/
│   ├── envs/
│   ├── nets/
│   └── utils/
├── experiments/
│   ├── imagenet/
│   ├── incremental_cifar/
│   ├── permuted_mnist/
│   ├── rl/
│   └── slowly_changing_regression/
├── configs/
│   ├── imagenet/
│   ├── incremental_cifar/
│   └── ... etc.
├── tests/
├── docs/
├── scripts/
├── data/
├── README.md
├── requirements.txt
└── setup.py