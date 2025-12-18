# Claude Context Summary

## Project Overview
Research codebase studying **learning dynamics and loss landscapes** in neural networks, with focus on:
- **Rank dynamics** during training (matrix ranks in layers)
- **Loss landscape smoothness** and optimization properties
- **Architectural variations** (normalized weights, layer norm, SVD decomposition)
- **Plasticity** in neural networks during task shifting
- **Continuous learning** algorithms like Generate-and-Test (GnT)
- **Rank-Restoring Continuous Backpropagation (RR-CBP)** - Recent major development for maintaining rank during continual learning

## Architecture Pattern
**Factory-based design** with clean separation:
- `src/models/` + `model_factory.py` - 21 model files, 17+ registered model types
- `src/algos/supervised/` + `supervised_factory.py` - Learning algorithms (backprop, CBP, RR-CBP, RR-GnT variants)
- `src/algos/RL/` + `rl_factory.py` - RL learners (PPO)
- `src/data_loading/` + `dataset_factory.py` - Dataset handling (CIFAR10, MNIST, ImageNet)
- `experiments/` - 19 experiment directories, 58 YAML configs, 9 train scripts
- `configs/configurations.py` - Type-safe dataclass configs (Hydra + OmegaConf)
- `src/utils/` - 18 utility modules (rank computation, weight tracking, analysis, Jacobian, Hessian)

## Key Commands
```bash
# Lint/typecheck (if available) - ask user for commands if not found
# Test running: cd experiments/<name> && python train.py
```

## Adding New Components

### New Model Architecture
1. Create `src/models/YourModel.py` (standard `nn.Module`)
2. Add branch in `src/models/model_factory.py:33+`:
   ```python
   if model_type == 'YourModel':
       if config.netparams is None:
           raise ValueError("config.params cannot be None for YourModel")
       return YourModel(config.netparams)
   ```
3. Add transforms in `src/data_loading/transform_factory.py`
4. Unit test: construction + forward pass

### New Learner/Optimizer  
1. Create `src/algos/supervised/your_learner.py` with:
   - `learn(input, target) -> (loss, output)` method
   - Set `self.previous_features` for rank tracking
2. Register in `src/algos/supervised/supervised_factory.py:12+`:
   ```python
   elif config.type == 'your_learner':
       return YourLearner(net, config, netconfig)
   ```

### New Dataset
- **Torchvision**: Set `DataConfig.dataset` to match class name
- **Custom**: Add branch in `dataset_factory.py:154+`

### New Experiment
1. Create `experiments/<name>/cfg/config.yaml` following `ExperimentConfig`
2. Create `experiments/<name>/train.py` with standard wiring:
   ```python
   @hydra.main(config_path="cfg", config_name="config")
   def main(cfg: ExperimentConfig):
       transform = transform_factory(cfg.data.dataset, cfg.net.type)
       train_set, test_set = dataset_factory(cfg.data, transform, with_testset=True)
       net = model_factory(cfg.net).to(cfg.net.device)
       learner = create_learner(cfg.learner, net, cfg.net)
       # ... training loop
   ```

## Configuration System (configs/configurations.py)

**Main Dataclasses:**
- `DataConfig` - Dataset configuration (dataset name, batch size, data paths)
- `NetParams` / `LinearNetParams` - Network architecture parameters
- `NetConfig` - Network configuration (type, device, parameters)
- `BaseLearnerConfig` - Base learner configuration
- `BackpropConfig` - Standard backpropagation config
- `ContinuousBackpropConfig` - CBP config (replacement frequency, utility threshold)
- `RRContinuousBackpropConfig` - RR-CBP v1 config
- `RRCBP2Config` - RR-CBP v2 config (use_energy_budget, orthogonalization settings)
- `EvaluationConfig` - Evaluation settings
- `LoggingConfig` - Logging configuration (wandb, local logging)
- `WandbConfig` - Weights & Biases integration
- `ExperimentConfig` - Top-level experiment config combining all above

Uses **Hydra** with **OmegaConf** for YAML-based configuration with runtime override support.

## Current Model Types (17+ registered)

**ConvNet Variants** (6): `ConvNet`, `ConvNet_norm`, `ConvNet_SVD`, `ConvNet_FC_layer_norm`, `ConvNet_conv_and_FC_layer_norm`, `ConvNet_batch_norm`

**VGG Variants** (2+): `vgg_custom`, `vgg_custom_norm` (vgg_normalized_conv.py recently modified)

**ResNet Variants** (3): `resnet_custom`, `full_rank_resnet_custom`, `resnet_normalized_conv`

**Deep FFNN Variants** (5): `deep_ffnn`, `deep_ffnn_weight_norm_single_rescale`, `deep_ffnn_weight_norm_multi_channel_rescale`, `deep_ffnn_weight_batch_norm`, `ffnn_normal_BN`

**RL Backbones** (2): `rl_cnn_backbone`, `rl_mlp_backbone`

**Additional components**: NormConv2d, NormalizedWeightsLinear, SVD-decomposed layers (conv/FC)

## Current Learner Types

**Basic Learners:**
- `backprop` - Standard backpropagation (basic_backprop.py)

**Continuous Backpropagation (CBP):**
- `continuous_backprop` - Periodically replaces low-utility mature neurons
- Variants: `ContinuousBackprop_for_ConvNet`, `ContinualBackprop_for_FC`
- With GnT support: `continuous_backprop_with_GnT.py`

**Rank-Restoring CBP (RR-CBP) - Version 1:**
- `rr_cbp` - Initializes new neurons with Σ-orthogonal weights to maintain rank
- Files: `rr_cbp_conv.py` (RankRestoringCBP_for_ConvNet), `rr_cbp_fc.py` (RankRestoringCBP_for_FC)

**Rank-Restoring CBP (RR-CBP2) - Version 2 (RECENT):**
- `rr_cbp2` - Unit Σ-norm variant (use_energy_budget=False)
- `rr_cbp_e_2` - Energy-budget controlled Σ-norm (use_energy_budget=True)
- Files: `rr_cbp2_conv.py`, `rr_cbp2_fc.py`
- **Supporting infrastructure** `src/algos/supervised/rank_restoring/`:
  - `sigma_geometry.py` - Σ-geometry operations with robust GPU/CPU fallback (recently enhanced)
  - `rr_covariance.py` - EMA covariance tracking
  - `rr_projection.py` - Σ-orthonormal basis operations
- **Key classes**: SigmaGeometry, SigmaProjector, EnergyAllocator, CovarianceState, SigmaOrthonormalBasis
- Documentation: 5 markdown files including algorithm guides and implementation analysis
- **Environment variable**: Set `SIGMA_FORCE_CPU_EIGH=1` to force CPU eigendecomposition for numerical stability

**Rank-Restoring GnT (RR-GnT):**
- `rr_gnt` variants (v1 and v2)
- Files: `rr_gnt_conv.py`, `rr_gnt_fc.py`, `rr_gnt2_conv.py`, `rr_gnt2_fc.py`

**Reinforcement Learning:**
- `ppo_learner` - Proximal Policy Optimization (ppo_learner.py)

**Optimizers:**
- AdamGnT - Adam with Grow-and-Trim (AdamGnT.py)
- Standard GnT implementation (gnt.py)

## Experiment Directories (19 total)

**Core Training:**
- `basic_training` - Standard supervised training
- `basic_RL` - Basic RL training

**Rank & Plasticity Studies:**
- `rank_tracking_in_shifting_task` - Rank tracking with task shifts
- `study_of_rank_tracking` - Rank tracking studies
- `rank_diminishing_property` - Rank diminishing research
- `comparison_of_different_measures_of_rank` - Rank measure comparisons
- `output_vs_input_plasticity` - Input vs output plasticity study
- `improving_plasticity_via_advanced_optimizer` - Plasticity optimization

**Loss Landscape & Hessian:**
- `Hessian_and_landscape_plot_with_plasticity_loss` - Loss landscape analysis suite
- `Hessian_and_loss_landscape_vs_plasticity` - Hessian analysis
- `normalization_loss_landscape` - Normalization effects on landscape
- `deep_linear_landscape` - Deep linear network landscape analysis

**Architectural Comparisons:**
- `Normalized_weights_vs_Batched_norm` - Normalization comparison
- `separating_depth_from_rank_reduction` - Depth vs rank study

**ImageNet & Task Shifting:**
- `shifting_tasks_on_Imagenet` - ImageNet task shifting

**Other:**
- `comparison_of_generalisation_properties_of_optimizers` - Optimizer comparison
- `hyperparameters_sweep` - Hyperparameter sweeps
- `tracking_in_RL` - RL rank tracking
- `_template` - Template for new experiments

## Dependencies (requirements.txt)

**Core ML:**
- torch (CPU version from pytorch.org)
- torchvision

**RL:**
- stable-baselines3[extra]
- gym
- box2d-py

**Configuration & Testing:**
- hydra-core==1.3.2
- omegaconf
- pyyaml
- pytest==8.3.4

**Scientific Computing:**
- scipy

**Environment:** Python with type hints, pytest for testing, CI/CD with conda

## Research Priorities (from README.md:9+)
- Layer norm models + effective learning rate
- Partial Jacobian rank support  
- Timer for rank computation performance
- Flexible data logging for raw data
- Multi-run experiment management
- Better plotting from raw data
- Rank diminishing research (partial rank, numerical rank, effective rank)

## Recent Development & Insights

**Latest Update: Enhanced SigmaGeometry Robustness (55a6e38)**
- **Most recent commit**: "Enhance SigmaGeometry robustness: fallback to CPU double-precision and increase regularization on failure"
- **Key improvements to `sigma_geometry.py`**:
  - Robust GPU eigendecomposition error handling (addresses cuSOLVER failures with ill-conditioned matrices)
  - CPU fallback mechanism with double-precision computation
  - Environment variable `SIGMA_FORCE_CPU_EIGH=1` to force CPU eigendecomposition
  - Matrix preconditioning: symmetry enforcement, regularization, condition number checks
  - Multiple recovery layers including diagonal approximation as last resort
- **Applied to**: SigmaGeometry, SigmaProjector, EnergyAllocator classes
- **Impact**: Significantly improved numerical stability and training reliability for RR-CBP2

**Major Feature: RR-CBP2 (Rank-Restoring Continuous Backpropagation v2)**
- Commits: "with rr cbp 2 analysis" (013d2da), "added rr cbp 2 implementation" (b1b3ee4)
- Two variants: rr_cbp2 (unit Σ-norm) and rr_cbp_e_2 (energy-budget controlled)
- **Supporting infrastructure** (`src/algos/supervised/rank_restoring/`):
  - `sigma_geometry.py` - Σ-geometry operations (SigmaGeometry, SigmaProjector, EnergyAllocator)
  - `rr_covariance.py` - Exponential moving average covariance tracking (CovarianceState)
  - `rr_projection.py` - Σ-orthonormal basis and projection operations
- **Comprehensive documentation**: 5 RR-CBP related markdown files with algorithm guides and implementation analysis

**Codebase Stats:**
- **58 YAML config files** across 19 experiment directories
- **21 model architecture files**, 17+ registered types
- **9 training scripts** across experiments
- **24 test files** for validation
- **18 utility modules** for analysis
- Factory pattern working well - architectural refactoring would be HIGH RISK
- Focus on research value over code elegance
- Additive improvements preferred over breaking changes

## Key Utilities (src/utils/ - 18 modules)

**Rank & Feature Analysis:**
- `first_order_extraction_measures.py` - First-order analysis (4,225 lines)
- `zeroth_order_features.py` - Zeroth-order feature extraction (30,307 lines)
- `partial_jacobian_source.py` - Partial Jacobian computation
- `jacobian_computations_comparison.py` - Jacobian computation methods
- `rank_drop_dynamics.py` - Rank dynamics tracking

**Loss Landscape:**
- `loss_landscape_smoothness_measures.py` - Smoothness metrics

**Weight Tracking:**
- `track_weights_norm.py` - Weight norm tracking
- `weight_norm_research_tools.py` - Weight norm analysis

**Infrastructure:**
- `data_logging.py` - Data logging infrastructure
- `feature_container.py` - Feature extraction container
- `task_shift_logging.py` - Task shift logging
- `gpu_health_check.py` - GPU monitoring
- `robust_checking_of_training_errors.py` - Error checking

**Other:**
- `plotting_tools.py` - Plotting utilities
- `research_based_investigation.py` - Research analysis tools
- `migration_example.py` - Code migration utilities
- `miscellaneous.py` - General utilities
- `sketch.py` - Experimental sketches

## Important Notes

**Best Practices:**
- **Reproducibility critical** - don't break existing experiments
- **Research velocity** > code architecture purity
- **Numerical stability**: For RR-CBP2, set `SIGMA_FORCE_CPU_EIGH=1` if encountering GPU eigendecomposition errors
- Uses Hydra configs extensively with dataclass validation

**Active Development:**
- Recent focus on RR-CBP2 robustness and numerical stability improvements
- Task shifting and rank tracking features under active development
- Error patterns: Many `ValueError` with config validation boilerplate

**Technical Details:**
- GPU eigendecomposition failures handled automatically with CPU fallback
- Matrix conditioning applied to prevent ill-conditioned covariance matrices
- Multiple recovery layers ensure training continues even with numerical issues