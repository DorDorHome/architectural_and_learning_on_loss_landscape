import torch

from configs.configurations import NetConfig, NetParams, RRContinuousBackpropConfig
from src.algos.supervised.rr_cbp_conv import RankRestoringCBP_for_ConvNet
from src.models.conv_net import ConvNet


def build_conv_model(device: str = "cpu"):
    params = NetParams(
        activation="relu",
        num_classes=5,
        input_height=28,
        input_width=28,
    )
    model = ConvNet(params)
    model = model.to(device)
    return model, params


def fake_conv_batch(device: str = "cpu"):
    torch.manual_seed(0)
    x = torch.randn(8, 3, 28, 28, device=device)
    y = torch.randint(0, 5, (8,), device=device)
    return x, y


def test_rank_restoring_conv_cbp_step_runs():
    device = "cpu"
    net, net_params = build_conv_model(device)
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
    config.opt = "adam"
    config.network_class = "conv"
    net_config = NetConfig(type="ConvNet", netparams=net_params, network_class="conv")

    learner = RankRestoringCBP_for_ConvNet(net, config, netconfig=net_config)

    x, y = fake_conv_batch(device)
    loss, _ = learner.learn(x, y)
    assert torch.isfinite(loss).item()

    loss2, _ = learner.learn(x, y)
    assert torch.isfinite(loss2).item()

    stats = learner.rr_gnt.get_layer_stats()
    assert len(stats) > 0
    total_replacements = sum(s.successful + s.fallbacks for s in stats.values())
    assert total_replacements >= 0
    for layer_stat in stats.values():
        if layer_stat.last_metrics is not None:
            assert "rank_WSigmaWT" in layer_stat.last_metrics
            assert "active_fraction" in layer_stat.last_metrics
