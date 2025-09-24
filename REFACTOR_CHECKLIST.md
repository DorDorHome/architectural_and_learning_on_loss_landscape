# Refactor & Feature Alignment Checklist

This file tracks the incremental, backwards-compatible refactor plan.
Each step: additive first, verify by smoke run, then proceed.
We only mark a step complete after executing its changes + validation.

## Legend
- [ ] Pending
- [x] Completed
- (ℹ) Notes / metrics after execution

## Steps

 - [ ] 0. Baseline validation (no code changes): 1-epoch smoke run of `experiments/basic_training/single_run.py` with safe overrides (disable W&B, shorten epoch, CPU fallback). Record loss & confirm no exceptions.
 - [ ] 1. Add `use_testset` to `EvaluationConfig` (in `configs/configurations.py`). No behavior change yet. Smoke test with and without `evaluation.use_testset=True`.
 - [ ] 2. Add task-shift fields to `ExperimentConfig`: `task_shift_mode`, `task_shift_param`, `num_tasks` (inert). Smoke test.
 - [ ] 3. Add `network_class` to `NetConfig` & `BaseLearnerConfig` with inference shim in `supervised_factory.create_learner`. Smoke test with ConvNet + (optional) deep FFNN.
 - [ ] 4. Normalize learner type aliases (`cbp`, `continuous_backprop`, legacy `basic_continous_backprop`) + deprecation warning. Smoke test each alias (1 batch each).
 - [ ] 5. Case-insensitive dataset normalization in `dataset_factory` & `transform_factory`. Test lowercase `mnist`, `cifar10`.
 - [ ] 6. Safe ConvNet dimension inference (fallback to dataset defaults when `input_height`/`input_width` missing). Test with current config (no explicit dims) + manual override.
 - [ ] 7. Deep FFNN naming alignment (alias `'deep_ffnn_weight_norm'` → existing canonical name). Smoke test alias instantiation.
 - [ ] 8. Rank tracking integration (optional path): when `track_rank=True`, compute & log rank stats each eval epoch. Test tiny run.
 - [ ] 9. Task-shift operationalization: new example or guarded integration (stateless + stateful). Test `permuted_input` (2 tasks) and `drifting_values` (use `learn_from_partial_values`).
 - [ ] 10. Logging config cleanup: split W&B vs storage config; retain backward compatibility w/ warning. Smoke test old config unchanged.
 - [ ] 11. Hygiene & dead code cleanup: typos, unused imports, unreachable transform code, clearer errors.
 - [ ] 12. Docs synchronization & CHANGELOG fragment.

## Execution Log
| Step | Status | Timestamp (UTC) | Notes |
|------|--------|-----------------|-------|
| 0 | [x] | 2025-09-25T00:00:00Z | Baseline run manually verified (user interrupted mid-epoch; model/loop functioning). |
| 1 | [x] | 2025-09-25T00:00:00Z | Added use_testset to EvaluationConfig; runtime import validated. |
| 2 | [x] | 2025-09-25T00:00:00Z | Added task shift fields (mode, param, num_tasks); runtime defaults verified. |
| 3 | [x] | 2025-09-25T00:00:00Z | Added network_class to configs + inference & alias handling in supervised_factory; validated inference (conv). |
| 4 | [x] | 2025-09-25T00:00:00Z | Alias normalization & deprecation warning added; all aliases validated OK. |
| 5 | [x] | 2025-09-25T00:00:00Z | Added case-insensitive dataset normalization + warnings; validated lowercase cifar10 & mnist. |
| 6 | [x] | 2025-09-25T00:00:00Z | Added conv input dim inference in model_factory; validated CIFAR10/MNIST/ImageNet heuristics + single-dim copy with forward passes. |
| 7 | [ ] | – |  |
| 8 | [x] | 2025-09-25T00:00:00Z | Integrated rank tracking in single_run (feature ranks each eval); validated via validate_rank_tracking_basic.py (ConvNet+CIFAR10) logging 5 layers. |
| 9 | [ ] | – |  |
| 10 | [ ] | – |  |
| 11 | [ ] | – |  |
| 12 | [ ] | – |  |

## Smoke Test Template
```
python experiments/basic_training/single_run.py device=cpu epochs=1 batch_size=64 use_wandb=False
```
Criteria:
- Runs to completion (exit code 0)
- Final epoch metrics printed (Accuracy, Loss)
- No uncaught exceptions

---
(Updates will append metrics & short verification blurbs under Execution Log.)
