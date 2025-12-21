ADR 005: Implementation of Grokking Experiments with RR-CBP Telemetry

1. Context

We aim to investigate the "Grokking" phenomenon (delayed generalization) through the lens of our existing Rank-Restoration Continuous Backprop (RR-CBP) research.
We need to implement the standard Modular Arithmetic task ($a + b \pmod p$) and a Transformer architecture, while ensuring full compatibility with the repository's existing telemetry (metrics for lambda_min, rank restoration, and orthogonality).

2. Design Philosophy

Modular Integration: New components must be implemented in standalone files and exposed only through existing factories (dataset_factory.py, model_factory.py).

Configuration Driven: All hyperparameters must be controlled via Hydra configs, backed by strongly-typed Dataclasses in configurations.py.

Telemetry Preservation: The model must support existing hooks. We require a "Manual" implementation of the Transformer to ensure internal weights ($W_Q, W_K, W_V$) are accessible to our rank-tracking hooks.

3. Detailed Specifications

3.1 Data Layer (src/data_loading/)

Requirement: Implement on-the-fly generation of the Modular Arithmetic dataset.

New File: src/data_loading/grokking_datasets.py

Class: ModularArithmeticDataset(torch.utils.data.Dataset)

Constructor: __init__(self, train: bool, seed: int = 42)

Logic:

Modulus ($p$): 113.

Operation: Addition.

Data Generation: 1. Generate list of all $p \times p$ pairs $(x, y)$.
2. Shuffle the list deterministically using seed.
3. Split: If train=True, keep first 50%. If train=False, keep remaining 50%.

Tokenization: * Indices $0 \dots p-1$: Numbers.

Index $p$: + token.

Index $p+1$: = token.

Input Tensor: [x, p, y, p+1].

Output: Returns (input_tensor, target_token) where target_token is $(x+y) \pmod p$.

Factory Update: src/data_loading/dataset_factory.py

Action: Add case for config.dataset == 'modular_arithmetic'.

Implementation: Instantiate ModularArithmeticDataset twice (once with train=True, once with train=False), passing a fixed seed (e.g., from config or default 42) to ensure disjoint sets.

3.2 Model Layer (src/models/)

Requirement: Implement two versions of the Transformer Decoder.

New File: src/models/grokking_transformer.py

Class A: GrokkingTransformerManual(nn.Module)

Design: Implement the decoder block using explicit nn.Linear layers for Attention (Q, K, V, O) and MLP.

Purpose: Allows RR-CBP telemetry hooks to easily access individual projection matrices.

Architecture: Pre-Norm, 1-2 Layers, GeLU, Learned Pos Embeddings.

Class B: GrokkingTransformerStandard(nn.Module)

Design: Use nn.TransformerDecoderLayer (ensure norm_first=True for Pre-Norm).

Purpose: Baseline/Verification using optimized PyTorch kernels.

Factory Update: src/models/model_factory.py

Logic:

If config.net.type == 'GrokkingTransformer_pytorch_manual_implementation': Instantiate GrokkingTransformerManual.

If config.net.type == 'GrokkingTransformer_pytorch_implementation': Instantiate GrokkingTransformerStandard.

3.3 Configuration Layer (configs/)

Requirement: Implement a dedicated configuration class and update the global type union.

File: configs/configurations.py

Action 1: Define GrokkingTransformerConfig dataclass.

Fields: vocab_size (115), max_seq_len (10), n_layers (2), n_heads (4), d_model (128), dropout (0.0).

Field type: str (default "GrokkingTransformer_pytorch_manual_implementation").

Action 2 (CRITICAL): Update the net field in the main config class (e.g., ExperimentConfig or NetParams parent) to use a Union.

Before: net: NetConfig = field(...)

After: net: Union[NetConfig, GrokkingTransformerConfig] = field(...)

Rationale: This ensures strict type checking allows our new config without hacking the existing NetConfig.

3.4 Experiment Configuration (experiments/)

Requirement: Create the run configuration.

New File: experiments/grokking/cfg/config.yaml

Defaults:

dataset: "modular_arithmetic"

net:

_target_: "configs.configurations.GrokkingTransformerConfig"

type: "GrokkingTransformer_pytorch_manual_implementation" # Default to manual for better telemetry

d_model: 128

n_layers: 2

learner: Standard BackpropConfig (or RRContinuousBackpropConfig).

Optimization: AdamW, lr=1e-3, weight_decay=1.0, epochs=10000.

3.5 Experiment Runner (experiments/grokking/)

Requirement: Create a dedicated training script that orchestrates the experiment and telemetry.

New File: experiments/grokking/train_grokking.py

Purpose: Main entry point for the Grokking experiment.

Responsibilities:

Initialize Hydra configuration from experiments/grokking/cfg/config.yaml.

Instantiate Data and Model using dataset_factory and model_factory.

Implement the training loop (epochs/steps).

Crucial: Calculate and log the telemetry (Rank, Loss, Acc) as defined in Section 4.7.

4. Implementation Steps for Agent

Create src/data_loading/grokking_datasets.py.

Update src/data_loading/dataset_factory.py.

Create src/models/grokking_transformer.py with both GrokkingTransformerManual and GrokkingTransformerStandard.

Update configs/configurations.py adding GrokkingTransformerConfig and updating the net Union type hint.

Update src/models/model_factory.py to handle the two new type strings.

Create experiments/grokking/cfg/config.yaml.

Create experiments/grokking/train_grokking.py (Main Script).

Initialize factories and run the training loop.

Note: This replaces reliance on a root-level train.py.

Telemetry Integration (Reference Pointer):

Goal: Log Rank Proxies (Effective Rank, Stable Rank, etc.).

Constraint: Do not reimplement the math.

Action: Locate the function compute_all_rank_measures_list (likely in src/utils or src/telemetry).

Reference: Check experiments/improving_plasticity_via_advanced_optimizer/train_with_improved_optimizer.py to see how this function is imported and called during the training loop.

Implementation: In experiments/grokking/train_grokking.py, ensure these metrics are computed and logged alongside Train/Test Loss/Acc.

5. Acceptance Criteria

Config Validity: python experiments/grokking/train_grokking.py executes without errors.

Model Selection: Changing config type to GrokkingTransformer_pytorch_implementation successfully swaps the backend class.

Telemetry: Logs show Train Loss, Test Loss, Train Acc, Test Acc AND the rank measures (e.g., effective_rank) defined in the reference code.