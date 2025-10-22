import torch
import pytest

from configs.configurations import RRContinuousBackpropConfig, LinearNetParams, NetConfig
from src.models.deep_ffnn import DeepFFNN
from src.algos.supervised.rr_cbp_fc import RankRestoringCBP_for_FC


def build_small_model(device="cpu"):
    net_params = LinearNetParams(input_size=8, num_features=6, num_outputs=3, num_hidden_layers=1, act_type='relu')
    model = DeepFFNN(net_params)
    model.type = 'FC'
    return model.to(device), net_params


def fake_batch(device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(16, 8, device=device)
    y = torch.randint(0, 3, (16,), device=device)
    return x, y


def test_rank_restoring_cbp_step_runs():
    device = 'cpu'
    net, net_params = build_small_model(device)
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=0.5,
        maturity_threshold=1,
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.0,
        sigma_ridge=1e-3,
        log_rank_metrics_every=1,
    )
    config.opt = 'adam'
    net_config = NetConfig(type='FC', netparams=net_params)

    learner = RankRestoringCBP_for_FC(net, config, netconfig=net_config)

    x, y = fake_batch(device)
    loss, _ = learner.learn(x, y)
    assert torch.isfinite(loss).item()

    loss2, _ = learner.learn(x, y)
    assert torch.isfinite(loss2).item()

    stats = learner.rr_gnt.get_layer_stats()
    assert len(stats) > 0
    total_replacements = sum(s.successful + s.fallbacks for s in stats.values())
    assert total_replacements >= 0
    for layer_stat in stats.values():
        assert layer_stat.last_metrics is not None
        assert 'rank_WSigmaWT' in layer_stat.last_metrics
        assert 'active_fraction' in layer_stat.last_metrics


@pytest.mark.parametrize("center_mode", ["mean", "median"])
def test_bias_centering(center_mode):
    device = 'cpu'
    net, net_params = build_small_model(device)
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=1.0,
        maturity_threshold=1,
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.0,
        center_bias=center_mode,
        sigma_ridge=1e-3,
        log_rank_metrics_every=1,
    )
    config.opt = 'adam'
    net_config = NetConfig(type='FC', netparams=net_params)

    learner = RankRestoringCBP_for_FC(net, config, netconfig=net_config)

    x, y = fake_batch(device)
    learner.learn(x, y)
    learner.learn(x, y)

    layer_bias = learner.net.layers[0].bias
    assert torch.isfinite(layer_bias).all()
