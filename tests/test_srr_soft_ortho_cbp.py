"""Tests for the isolated-flow SRR-CBP variants (`srr_aso_cbp`, `srr_faso_cbp`).

Covers:
  1. Gradient-isolation regression test (the bug the new learners exist to fix).
  2. Hand-computed correctness of `compute_aso_loss`.
  3. FASO routing equivalence (Approach A vs Approach B).
  4. Config validation for ASO vs FASO discriminator.
  5. ``include_bias_in_A_reg`` toggle behaviour.
  6. End-to-end ``learn()`` smoke test on FC and Conv for both type values.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from configs.configurations import (
    LinearNetParams,
    SRRSoftOrthoCBPConfig,
)
from src.algos.supervised.age_weighted_so_losses import (
    _faso_NxN_branch,
    _faso_mxm_branch,
    compute_aso_loss,
    compute_faso_loss,
    faso_routing_branch,
)
from src.algos.supervised.supervised_factory import create_learner
from src.algos.supervised.srr_soft_ortho_cbp import (
    SRR_SoftOrtho_CBP_for_ConvNet,
    SRR_SoftOrtho_CBP_for_FC,
)
from src.models.conv_net import ConvNet
from src.models.deep_ffnn import DeepFFNN


# ---------------------------------------------------------------------------
# Fixtures / config builders
# ---------------------------------------------------------------------------


def _dummy_conv_net_params():
    return OmegaConf.create({
        'num_classes': 10,
        'input_height': 32,
        'input_width': 32,
        'activation': 'relu',
    })


def _dummy_fc_net_params():
    return LinearNetParams(
        input_size=32 * 32 * 3,
        num_features=64,
        num_outputs=10,
        num_hidden_layers=2,
        act_type='relu',
        initialization='kaiming',
    )


def _aso_config(**overrides) -> SRRSoftOrthoCBPConfig:
    base = dict(
        type='srr_aso_cbp',
        device='cpu',
        opt='adam',
        step_size=0.01,
        SO_reg_lambda=0.1,
        age_decay_rate=0.99,
        normalization_mode="naive mse sum correction",
        aso_maturity_threshold=10,
        maturity_threshold=10,
        neurons_replacement_rate=0.1,
        util_type='contribution',
        include_bias_in_A_reg=False,
    )
    base.update(overrides)
    return SRRSoftOrthoCBPConfig(**base)


def _faso_config(**overrides) -> SRRSoftOrthoCBPConfig:
    base = dict(
        type='srr_faso_cbp',
        device='cpu',
        opt='adam',
        step_size=0.01,
        SO_reg_lambda=0.1,
        age_decay_rate=0.99,
        normalization_mode="naive mse sum correction",
        aso_maturity_threshold=None,
        maturity_threshold=10,
        neurons_replacement_rate=0.1,
        util_type='contribution',
        include_bias_in_A_reg=False,
    )
    base.update(overrides)
    return SRRSoftOrthoCBPConfig(**base)


def _make_fc_model():
    params = _dummy_fc_net_params()
    model = DeepFFNN(params)
    model.type = 'FC'  # mirrors the convention used in tests/test_srr_cbp.py
    net_config = OmegaConf.create({'type': 'FC', 'netparams': params})
    return model, net_config


def _make_conv_model():
    params = _dummy_conv_net_params()
    model = ConvNet(params)
    net_config = OmegaConf.create({'type': 'ConvNet', 'netparams': params})
    return model, net_config


# ---------------------------------------------------------------------------
# 1. Gradient-isolation regression test
# ---------------------------------------------------------------------------


def test_aso_loss_does_not_leak_gradient_to_earlier_layers():
    """`compute_aso_loss(F.linear(H.detach(), W2), ...)` must not generate
    gradients for the layer that produced ``H``. This is precisely the bug
    that the old SRR-CBP suffered from (it used the forward output of
    layer-2, leaving the graph through ``H`` open).
    """
    torch.manual_seed(0)
    layer1 = nn.Linear(8, 6)
    layer2 = nn.Linear(6, 5)
    x = torch.randn(4, 8)

    # Forward through the first layer; H carries gradient back to layer1.
    H = layer1(x)
    assert H.requires_grad

    # Spec contract: use stop_gradient(H) when computing A_reg.
    A_reg = F.linear(H.detach(), layer2.weight, bias=None)  # [B=4, N=5]
    A_reg = A_reg.t()                                       # [N=5, m=4]

    ages = torch.tensor([0, 0, 10, 10, 10], dtype=torch.long)
    loss = compute_aso_loss(
        A_reg=A_reg,
        ages=ages,
        maturity_threshold=2,
        age_decay_rate=0.9,
        normalization_mode="no correction",
    )

    # Make sure all params start without grads.
    for p in list(layer1.parameters()) + list(layer2.parameters()):
        assert p.grad is None

    loss.backward()

    # Layer-1 must be completely isolated (no path from loss to it).
    assert layer1.weight.grad is None or torch.all(layer1.weight.grad == 0.0)
    assert layer1.bias.grad is None or torch.all(layer1.bias.grad == 0.0)

    # Layer-2 must receive a non-trivial gradient on its weight rows
    # corresponding to the young units (indices 0 and 1).
    assert layer2.weight.grad is not None
    young_rows = layer2.weight.grad[:2]
    mature_rows = layer2.weight.grad[2:]
    assert not torch.all(young_rows == 0.0), "Young units did not receive ASO gradient"
    assert torch.all(mature_rows == 0.0), "Mature units must be stop-gradient'd"

    # `bias=None` -> layer-2 bias must be unaffected.
    assert layer2.bias.grad is None or torch.all(layer2.bias.grad == 0.0)


# ---------------------------------------------------------------------------
# 2. ASO mathematical correctness (hand computation)
# ---------------------------------------------------------------------------


def test_compute_aso_loss_matches_hand_calculation():
    """Compute the ASO loss by hand for a small fixed tensor and compare."""
    torch.manual_seed(123)
    A_reg = torch.randn(4, 5, dtype=torch.float64)  # N=4, m=5
    ages = torch.tensor([0, 1, 10, 10], dtype=torch.long)
    gamma = 0.9
    M = 2

    # Manual reference: indices 0, 1 are young; indices 2, 3 are mature.
    A_mature = A_reg[[2, 3], :].detach()
    raw = 0.0
    for y, age in [(0, 0), (1, 1)]:
        row = A_reg[y] @ A_mature.t()  # length-K vector
        raw += (gamma ** age) * (row * row).sum().item()
    expected_no_corr = raw / (2 * 5 * 5)
    Y, K = 2, 2
    expected_input_corr = expected_no_corr / Y
    expected_pair_corr = expected_no_corr / (Y * K)

    got_no = compute_aso_loss(A_reg, ages, M, gamma, "no correction")
    got_in = compute_aso_loss(A_reg, ages, M, gamma, "correct by input size")
    got_pr = compute_aso_loss(A_reg, ages, M, gamma, "naive mse sum correction")

    assert math.isclose(got_no.item(), expected_no_corr, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(got_in.item(), expected_input_corr, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(got_pr.item(), expected_pair_corr, rel_tol=1e-10, abs_tol=1e-12)


def test_compute_aso_loss_empty_partitions_return_zero():
    """If all units are young or all mature, the loss is exactly zero."""
    A_reg = torch.randn(4, 5)
    all_young = torch.zeros(4, dtype=torch.long)
    all_mature = torch.full((4,), 100, dtype=torch.long)
    loss_young = compute_aso_loss(A_reg, all_young, 10, 0.9, "no correction")
    loss_mature = compute_aso_loss(A_reg, all_mature, 10, 0.9, "no correction")
    assert loss_young.item() == 0.0
    assert loss_mature.item() == 0.0


# ---------------------------------------------------------------------------
# 3. FASO routing equivalence
# ---------------------------------------------------------------------------


def test_faso_routing_branch_dispatch_rule():
    assert faso_routing_branch(4, 8) == "NxN"   # N < m
    assert faso_routing_branch(8, 4) == "mxm"   # N > m
    assert faso_routing_branch(4, 4) == "mxm"   # N == m -> Approach B per spec


def test_faso_two_branches_agree_numerically():
    """For a single fixed ``A_reg``, both Approach A and Approach B must
    produce the same off-diagonal age-weighted sum (modulo float).
    """
    torch.manual_seed(7)
    # Use float64 to keep the agreement tight.
    A_reg = torch.randn(6, 6, dtype=torch.float64)
    ages = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.long)
    gamma = 0.85

    A_tilde = A_reg.detach()
    lambda_w = gamma ** ages.to(A_reg.dtype)

    loss_NxN = _faso_NxN_branch(A_reg, A_tilde, lambda_w)
    loss_mxm = _faso_mxm_branch(A_reg, A_tilde, lambda_w)
    assert torch.allclose(loss_NxN, loss_mxm, rtol=1e-10, atol=1e-12), (
        f"FASO branches disagree: NxN={loss_NxN.item():.12e}, "
        f"mxm={loss_mxm.item():.12e}"
    )


def test_faso_compute_loss_matches_NxN_direct_for_both_shapes():
    """Cross-check `compute_faso_loss` against a direct hand computation
    for both shape regimes (``N < m`` triggers the NxN branch, ``N >= m``
    triggers the mxm branch); the resulting loss must equal the explicit
    off-diagonal computation in either case.
    """
    torch.manual_seed(11)
    gamma = 0.95
    mode = "no correction"

    for N, m in [(4, 8), (8, 4)]:
        A_reg = torch.randn(N, m, dtype=torch.float64)
        ages = torch.arange(N, dtype=torch.long)
        loss = compute_faso_loss(A_reg, ages, gamma, mode)

        # Reference: explicit double loop over (i, j != i).
        A_t = A_reg.detach()
        lam = gamma ** ages.to(A_reg.dtype)
        ref = 0.0
        for i in range(N):
            s = 0.0
            for j in range(N):
                if j == i:
                    continue
                s += float((A_reg[i] @ A_t[j]).item() ** 2)
            ref += lam[i].item() * s
        ref = ref / (2 * m * m)

        assert math.isclose(loss.item(), ref, rel_tol=1e-9, abs_tol=1e-12), (
            f"FASO mismatch at (N={N}, m={m}): got={loss.item():.12e}, ref={ref:.12e}"
        )


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------


def test_config_validation_faso_rejects_aso_maturity_threshold():
    model, net_config = _make_fc_model()
    bad_cfg = _faso_config(aso_maturity_threshold=10)
    with pytest.raises(ValueError, match="srr_faso_cbp"):
        SRR_SoftOrtho_CBP_for_FC(model, bad_cfg, net_config)


def test_config_validation_aso_requires_threshold():
    model, net_config = _make_fc_model()
    bad_cfg = _aso_config(aso_maturity_threshold=None)
    with pytest.raises(ValueError, match="srr_aso_cbp"):
        SRR_SoftOrtho_CBP_for_FC(model, bad_cfg, net_config)


def test_config_validation_unknown_type():
    model, net_config = _make_fc_model()
    bad_cfg = _aso_config()
    bad_cfg.type = 'bogus_type'
    with pytest.raises(ValueError, match="Unsupported type"):
        SRR_SoftOrtho_CBP_for_FC(model, bad_cfg, net_config)


def test_config_round_trip_construction_succeeds_for_aso_and_faso():
    # ASO: positive maturity threshold.
    model_a, net_config_a = _make_fc_model()
    learner_a = SRR_SoftOrtho_CBP_for_FC(model_a, _aso_config(), net_config_a)
    assert callable(learner_a._regularize)
    assert learner_a._regularize is not None

    # FASO: aso_maturity_threshold left as None.
    model_f, net_config_f = _make_fc_model()
    learner_f = SRR_SoftOrtho_CBP_for_FC(model_f, _faso_config(), net_config_f)
    assert callable(learner_f._regularize)


# ---------------------------------------------------------------------------
# 5. include_bias_in_A_reg toggle behaviour
# ---------------------------------------------------------------------------


def _isolated_regularization_grads(learner, x):
    """Manually replicate the regularization-only backward path of the
    learner so we can inspect biases without the optimizer touching them.

    Returns a dict ``{param_name: grad_tensor}`` containing only the
    parameters that received non-None gradients from the regularization.
    """
    learner.layer_inputs = {}
    output, _features = learner.net.predict(x)
    # Zero existing grads on the network.
    for p in learner.net.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    reg_loss = torch.tensor(0.0, device=learner.device)
    for i in range(len(learner.gnt.ages)):
        entry = learner.layer_inputs.get(i)
        if entry is None:
            continue
        module, H = entry
        A_reg = learner._compute_A_reg(module, H)
        if A_reg is None:
            continue
        reg_loss = reg_loss + learner._regularize(A_reg, learner.gnt.ages[i])

    # Make backward unconditionally well-defined even if reg_loss == 0.
    if not reg_loss.requires_grad:
        reg_loss = reg_loss + 0.0 * sum(p.sum() for p in learner.net.parameters())
    reg_loss.backward()

    grads = {}
    for name, p in learner.net.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone()
    return grads, output


def test_include_bias_in_A_reg_toggle_controls_bias_gradient():
    torch.manual_seed(0)
    x = torch.randn(4, 32 * 32 * 3)
    target = torch.randint(0, 10, (4,))

    # --- include_bias=False: bias must receive zero grad from reg path -----
    model_off, net_config_off = _make_fc_model()
    cfg_off = _aso_config(SO_reg_lambda=1.0, include_bias_in_A_reg=False)
    learner_off = create_learner(cfg_off, model_off, net_config_off)
    # Force a mix of young/mature so the regularizer is non-trivial.
    for i in range(len(learner_off.gnt.ages)):
        n = learner_off.gnt.ages[i].shape[0]
        learner_off.gnt.ages[i][:n // 2] = cfg_off.aso_maturity_threshold + 1
        learner_off.gnt.ages[i][n // 2:] = 0

    grads_off, _ = _isolated_regularization_grads(learner_off, x)
    bias_off_names = [n for n in grads_off if n.endswith('.bias')]
    for name in bias_off_names:
        assert torch.all(grads_off[name] == 0.0), (
            f"include_bias_in_A_reg=False: bias {name} got a non-zero reg gradient"
        )

    # --- include_bias=True: at least one hidden bias must get a non-zero grad
    model_on, net_config_on = _make_fc_model()
    cfg_on = _aso_config(SO_reg_lambda=1.0, include_bias_in_A_reg=True)
    learner_on = create_learner(cfg_on, model_on, net_config_on)
    for i in range(len(learner_on.gnt.ages)):
        n = learner_on.gnt.ages[i].shape[0]
        learner_on.gnt.ages[i][:n // 2] = cfg_on.aso_maturity_threshold + 1
        learner_on.gnt.ages[i][n // 2:] = 0

    grads_on, _ = _isolated_regularization_grads(learner_on, x)
    bias_on_names = [n for n in grads_on if n.endswith('.bias')]
    any_nonzero = any(torch.any(grads_on[name] != 0.0) for name in bias_on_names)
    assert any_nonzero, (
        "include_bias_in_A_reg=True must produce at least one non-zero bias grad "
        "from the regularization path"
    )

    # --- Sanity: full learn() step still trains biases via task loss in both
    # configurations (the toggle must only affect the side-computation of A_reg).
    for cfg_builder, builder_name in [(_aso_config, 'False'), (lambda **kw: _aso_config(include_bias_in_A_reg=True, **kw), 'True')]:
        model_full, net_config_full = _make_fc_model()
        learner_full = create_learner(cfg_builder(), model_full, net_config_full)
        # Snapshot biases.
        bias_before = {
            name: p.detach().clone()
            for name, p in model_full.named_parameters() if name.endswith('.bias')
        }
        learner_full.learn(x, target)
        bias_after = {
            name: p.detach().clone()
            for name, p in model_full.named_parameters() if name.endswith('.bias')
        }
        # At least one bias should have moved (driven by the task loss).
        moved = any(
            not torch.allclose(bias_before[n], bias_after[n])
            for n in bias_before
        )
        assert moved, (
            f"Biases never moved during learn() with include_bias_in_A_reg={builder_name}; "
            "the forward pass appears to be incorrectly affected by the toggle."
        )


# ---------------------------------------------------------------------------
# 6. End-to-end smoke tests for both type values on FC and Conv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("learner_type", ["srr_aso_cbp", "srr_faso_cbp"])
def test_learn_step_smoke_fc(learner_type):
    torch.manual_seed(0)
    model, net_config = _make_fc_model()
    cfg = (_aso_config() if learner_type == 'srr_aso_cbp' else _faso_config())
    learner = create_learner(cfg, model, net_config)

    x = torch.randn(4, 32 * 32 * 3)
    target = torch.randint(0, 10, (4,))
    loss, output = learner.learn(x, target)

    assert loss is not None
    assert output.shape == (4, 10)
    assert torch.isfinite(loss).all()
    # Forward hooks must have populated layer_inputs.
    assert len(learner.layer_inputs) > 0
    for i, (module, H) in learner.layer_inputs.items():
        assert isinstance(module, (nn.Linear, nn.Conv2d))
        assert H is not None


@pytest.mark.parametrize("learner_type", ["srr_aso_cbp", "srr_faso_cbp"])
def test_learn_step_smoke_conv(learner_type):
    torch.manual_seed(0)
    model, net_config = _make_conv_model()
    cfg = (_aso_config() if learner_type == 'srr_aso_cbp' else _faso_config())
    learner = create_learner(cfg, model, net_config)

    x = torch.randn(4, 3, 32, 32)
    target = torch.randint(0, 10, (4,))
    loss, output = learner.learn(x, target)

    assert loss is not None
    assert output.shape == (4, 10)
    assert torch.isfinite(loss).all()
    assert len(learner.layer_inputs) > 0
