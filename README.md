# architectural_and_learning_on_loss_landscape

# to-do:
# layer norm models need to pair with use of effective learning rate 
# add support for partial jacabian rank.
# add timer for rank computation to assess computational bottleneck.



### create flexible way to manage data logging for raw data
### manage experiments with many runs, one after another, save to different folders under different names
### create flexible way to plot from raw data
### 



#### replicate "rank diminising in deep neural networks"
- subtasks:
- function for partial rank calculation
- function for numerical rank
- function for effective rank


#### Refer to "How does batch normalization help optimization", 
- subtask: mechanism/function for measure change in loss (Lipschitzness of loss function)
- subtask: mechanism/function for measuring gradient predictiveness ("beta"-smoothness of loss function, or Lipschitzness of gradient of loss function)
- Then, test it on both VGG and DLN.

#### new architecture:
##### Full-rank projection network:
Instead of mapping each layer to a lower dimension, map it to the same dimension, followed by a projection, then add the constant.
##### decomposed normalized CNN:
- for each layer, with normalized weights, compute normalized weights, then obtained an normlized weights. Used that for forward pass. 
(efficiency? Reference layer norm implementation for idea.)
--- update normalized_weights_FC with better track of variance of input
--- how should input variance be related to the best weight norm for best loss landscape? 
- finally, multiply a constant factor to each filter with normalized weights.
-- separate behavior of .train() vs .
- calculate the time of forward/backward pass compared to baseline.
#### parametric non-linear relu:
- use parametric relu
- use MaxOut as activation.

### SVD decomposition parametrisation layer:
Use SVD decomposition, with a non-linear map sandwiched between.



## handling loss landscape of layer dependency:
#### main idea: Invent algorithms for having different learning rates for different layers 
- use and exponentially decaying version, with faster decay at earlier layers, vs faster decay at later layers.
- updates later layers more frequently
- higher learning rates for later layers





write model loading (done)
- model factory, outline from chatgpt
- use VGG and convnet first
- test model loading.

- write backprop (done!)
- test_backprop.py (done)
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