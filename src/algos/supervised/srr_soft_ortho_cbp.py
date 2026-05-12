"""Soft Rank-Restoring Continual Backprop with isolated-flow regularizers.

Provides two user-facing learner types (`srr_aso_cbp`, `srr_faso_cbp`)
backed by one implementation per topology:

    - SRR_SoftOrtho_CBP_for_FC
    - SRR_SoftOrtho_CBP_for_ConvNet

Both classes capture each hidden layer's INPUT via forward hooks and, in
`learn()`, recompute the regularization preactivations as
``A_reg = W @ stop_gradient(H)`` (via ``F.linear`` / ``F.conv2d`` with
``bias`` defaulting to ``None``). This fixes the gradient-leakage bug in
`src/algos/supervised/srr_cbp.py`, where the output-hook variant lets
gradients flow from the ASO penalty into earlier layers through ``H``.

The choice of regularizer (ASO vs FASO) is bound once at construction
time from ``config.type`` and dispatched without further branching in
the training loop.
"""

from typing import Callable, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algos.AdamGnT import AdamGnT
from src.algos.gnt import ConvGnT_for_ConvNet, GnT_for_FC
from src.algos.supervised.age_weighted_so_losses import (
    _canonical_normalization_mode,
    compute_aso_loss,
    compute_faso_loss,
)
from src.algos.supervised.base_learner import Learner
from configs.configurations import NetConfig, SRRSoftOrthoCBPConfig


_RegularizerFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _make_regularizer(config: SRRSoftOrthoCBPConfig) -> _RegularizerFn:
    """Bind a single regularizer once based on ``config.type``.

    Raises ValueError for unsupported types and for ASO/FASO-specific
    config-validation failures (so we fail fast at construction time
    rather than in the hot path).
    """
    normalization_mode = _canonical_normalization_mode(config.normalization_mode)

    if config.type == 'srr_aso_cbp':
        if not (isinstance(config.aso_maturity_threshold, int)
                and config.aso_maturity_threshold > 0):
            raise ValueError(
                "srr_aso_cbp requires aso_maturity_threshold to be a positive int; "
                f"got {config.aso_maturity_threshold!r}."
            )
        maturity_threshold = int(config.aso_maturity_threshold)
        gamma = float(config.age_decay_rate)

        def _regularize_aso(A_reg: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
            return compute_aso_loss(
                A_reg, ages, maturity_threshold, gamma, normalization_mode
            )

        return _regularize_aso

    if config.type == 'srr_faso_cbp':
        if config.aso_maturity_threshold is not None:
            raise ValueError(
                "srr_faso_cbp does not use aso_maturity_threshold; leave it as None "
                "(or omit it). The GnT-side maturity_threshold is independent."
            )
        gamma = float(config.age_decay_rate)

        def _regularize_faso(A_reg: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
            return compute_faso_loss(A_reg, ages, gamma, normalization_mode)

        return _regularize_faso

    raise ValueError(
        f"Unsupported type for SRR_SoftOrtho_CBP: {config.type!r}. "
        "Expected 'srr_aso_cbp' or 'srr_faso_cbp'."
    )


def _extract_hidden_activation(netparams) -> str:
    if netparams is None:
        raise ValueError(
            "hidden_activation must be specified in netparams.activation or "
            "netparams.act_type (netparams=None)"
        )
    for attr_name in ('activation', 'act_type'):
        if hasattr(netparams, attr_name):
            value = getattr(netparams, attr_name)
            if value is not None:
                return value
    raise ValueError(
        f"hidden_activation must be specified in netparams.activation or "
        f"netparams.act_type (netparams={netparams})"
    )


class _SRRSoftOrthoCBPBase(Learner):
    """Shared body for the FC and Conv flavours of SRR-SoftOrtho-CBP.

    Subclasses are responsible for:
      - Constructing the appropriate GnT (`GnT_for_FC` vs
        `ConvGnT_for_ConvNet`).
      - Enforcing the topology check on ``net.type``.

    Everything else - hook registration, ``A_reg`` recomputation,
    optimizer plumbing, regularizer dispatch, GnT updates - lives here.
    """

    # ----- Class-level attribute annotations -------------------------------
    # `self.gnt` is constructed inside each subclass `__init__` (after
    # `super().__init__`) because the GnT class differs by topology. Declaring
    # the type here gives static analysers and IDEs the right interface for
    # the base-class methods that consume `self.gnt.ages` and
    # `self.gnt.gen_and_test(...)`. `GnT_for_FC` and `ConvGnT_for_ConvNet`
    # share no common ancestor (both inherit directly from `object`), so a
    # Union is the most accurate description we can give.
    gnt: Union[GnT_for_FC, ConvGnT_for_ConvNet]
    # Bound at construction time by `_make_regularizer`.
    _regularize: _RegularizerFn
    # Per-layer cache populated by forward hooks during `learn()`.
    layer_inputs: dict

    def __init__(
        self,
        net: nn.Module,
        config: SRRSoftOrthoCBPConfig,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        netparams = netconfig.netparams if netconfig is not None else None
        super().__init__(net, config, netparams)

        # Bind the regularizer & run config validation first so we fail
        # fast before building the optimizer / GnT.
        self._regularize = _make_regularizer(config)

        self.SO_reg_lambda = float(config.SO_reg_lambda)
        self.age_decay_rate = float(config.age_decay_rate)
        self.normalization_mode = _canonical_normalization_mode(config.normalization_mode)
        self.include_bias_in_A_reg = bool(config.include_bias_in_A_reg)

        self.neurons_replacement_rate = config.neurons_replacement_rate
        self.decay_rate_utility_track = config.decay_rate_utility_track
        self.maturity_threshold = config.maturity_threshold  # GnT-side only
        self.util_type = config.util_type
        self.init = config.init
        self.accumulate = config.accumulate
        self.outgoing_random = config.outgoing_random

        self.use_grad_clip = bool(config.use_grad_clip)
        self.grad_clip_max_norm = float(config.grad_clip_max_norm)

        if config.opt == 'adam':
            self.opt = AdamGnT(
                self.net.parameters(),
                lr=config.step_size,
                betas=(config.beta_1, config.beta_2),
                weight_decay=float(config.weight_decay),
            )

        self._netparams = netparams
        # Per-layer cache: layer_idx -> (module, input_tensor H).
        self.layer_inputs = {}

    # --- Hook plumbing -----------------------------------------------------

    def _hidden_weight_modules(self) -> list:
        """Return the list of weight modules whose INPUT we want to capture.

        Mirrors the iteration in `srr_cbp.py` (hidden layers only - the
        final classifier is excluded so the layer enumeration aligns with
        ``self.gnt.ages``).

        Two control-flow branches are mutually exclusive:
        1. **Plasticity-map path (preferred):** when the network exposes
           ``get_plasticity_map``, use it as the source of truth and skip
           the final classifier entry.
        2. **Legacy fallback:** for older models without
           ``get_plasticity_map`` (kept for backward compatibility with
           the `srr_cbp.py` enumeration), walk ``net.layers`` at even
           indices and clip to ``len(self.gnt.ages)`` so the layer
           enumeration aligns with the GnT's per-layer state.
        """
        if hasattr(self.net, "get_plasticity_map"):
            # Branch 1: plasticity-map path (preferred for migrated models).
            plasticity_map = self.net.get_plasticity_map()
            return [
                item['weight_module']
                for i, item in enumerate(plasticity_map)
                if i < len(plasticity_map) - 1
            ]
        else:
            # Branch 2: legacy fallback for nets without a plasticity map.
            layers = cast(Sequence[nn.Module], self.net.layers)
            modules: list = []
            for i in range(0, len(layers), 2):
                if isinstance(layers[i], (nn.Conv2d, nn.Linear)):
                    modules.append(layers[i])
            # The legacy walk historically also captured the final layer;
            # mimic `srr_cbp.py` exactly by limiting to ``len(self.gnt.ages)``
            # entries so the indexing stays aligned with the GnT state.
            max_layers = len(self.gnt.ages)
            return modules[:max_layers]

    def _register_hooks(self) -> None:
        modules = self._hidden_weight_modules()

        def make_hook(layer_idx: int, module: nn.Module):
            def hook(_module, input_tuple, _output):
                # `input_tuple` is a tuple; the first entry is the input
                # tensor fed into this layer.
                if input_tuple is None or len(input_tuple) == 0:
                    return
                H = input_tuple[0]
                self.layer_inputs[layer_idx] = (module, H)
            return hook

        for i, module in enumerate(modules):
            if i >= len(self.gnt.ages):
                break
            module.register_forward_hook(make_hook(i, module))

    # --- A_reg recomputation -----------------------------------------------

    def _compute_A_reg(self, module: nn.Module, H: torch.Tensor) -> Optional[torch.Tensor]:
        """Recompute the layer's preactivations with the input detached.

        Returns ``A_reg`` of shape ``[N, m]`` (units x effective batch),
        or ``None`` if the module type is unsupported.

        Default (``include_bias_in_A_reg=False``) matches the spec
        formulation ``A = W H``. Setting the flag to True opts into a
        bias-inclusive variant that *also* produces a gradient w.r.t.
        ``module.bias`` from the regularization path. Either way, the
        model's normal forward pass (used for the task loss, evaluation,
        and GnT feature statistics) continues to use the module with its
        bias, so this only changes the side-computation of ``A_reg``.
        """
        H_det = H.detach()

        if isinstance(module, nn.Linear):
            bias = module.bias if self.include_bias_in_A_reg else None
            out = F.linear(H_det, module.weight, bias)
            # If H_det was [B, in] -> out is [B, N]. If H_det had extra
            # leading dims (e.g. [B, T, in]), flatten them into the
            # batch dimension so A_reg ends up [N, m] with m = B*T*...
            if out.dim() < 2:
                return None
            if out.dim() == 2:
                return out.t()  # [N, B]
            # >=3 dims: collapse all leading dims into m.
            flat = out.reshape(-1, out.shape[-1])  # [B*..., N]
            return flat.t()  # [N, B*...]

        if isinstance(module, nn.Conv2d):
            bias = module.bias if self.include_bias_in_A_reg else None
            out = F.conv2d(
                H_det,
                module.weight,
                bias,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )  # [B, C, H', W']
            B, C, Hh, Ww = out.shape
            return out.transpose(0, 1).reshape(C, -1)  # [N=C, m=B*H'*W']

        return None

    # --- Training step -----------------------------------------------------

    def learn(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = x.to(self.device), target.to(self.device)

        # Clear stale captures from prior steps so we never accidentally
        # regularize against an old graph (especially relevant if a layer
        # was unused in the most recent forward, e.g., conditional nets).
        self.layer_inputs = {}

        output, features = self.net.predict(x)
        task_loss = self.loss_func(output, target)

        reg_loss = torch.tensor(0.0, device=self.device)

        for i in range(len(self.gnt.ages)):
            entry = self.layer_inputs.get(i)
            if entry is None:
                continue
            module, H = entry
            A_reg = self._compute_A_reg(module, H)
            if A_reg is None:
                continue

            layer_loss = self._regularize(A_reg, self.gnt.ages[i])
            reg_loss = reg_loss + layer_loss

        reg_loss = self.SO_reg_lambda * reg_loss
        total_loss = task_loss + reg_loss

        self.opt.zero_grad()
        total_loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_max_norm)

        self.opt.step()

        self.opt.zero_grad()
        self.gnt.gen_and_test(features=features)

        return task_loss.detach(), output.detach()


class SRR_SoftOrtho_CBP_for_FC(_SRRSoftOrthoCBPBase):
    """SRR Soft-Orthogonality CBP for fully-connected networks.

    The regularizer (ASO or FASO) is selected from ``config.type``.
    """

    def __init__(
        self,
        net: nn.Module,
        config: SRRSoftOrthoCBPConfig,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        super().__init__(net, config, netconfig)

        if getattr(self.net, 'type', None) != 'FC':
            raise TypeError(
                f"SRR_SoftOrtho_CBP_for_FC requires net.type == 'FC', "
                f"got {getattr(self.net,'type',None)}"
            )

        hidden_activation = _extract_hidden_activation(self._netparams)

        self.gnt = GnT_for_FC(
            net=self.net,
            hidden_activation=hidden_activation,
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            maturity_threshold=self.maturity_threshold,
            util_type=self.util_type,
            device=self.device,
            loss_func=self.loss_func,
            init=self.init,
            accumulate=self.accumulate,
        )

        self._register_hooks()


class SRR_SoftOrtho_CBP_for_ConvNet(_SRRSoftOrthoCBPBase):
    """SRR Soft-Orthogonality CBP for ConvNet-style topologies.

    The regularizer (ASO or FASO) is selected from ``config.type``.
    """

    def __init__(
        self,
        net: nn.Module,
        config: SRRSoftOrthoCBPConfig,
        netconfig: Optional[Union[NetConfig, None]] = None,
    ):
        super().__init__(net, config, netconfig)

        hidden_activation = _extract_hidden_activation(self._netparams)

        num_last_filter_outputs = self._calculate_last_filter_outputs()

        self.gnt = ConvGnT_for_ConvNet(
            net=self.net,
            hidden_activation=hidden_activation,
            opt=self.opt,
            replacement_rate=self.neurons_replacement_rate,
            decay_rate=self.decay_rate_utility_track,
            init=self.init,
            num_last_filter_outputs=num_last_filter_outputs,
            util_type=self.util_type,
            maturity_threshold=self.maturity_threshold,
            device=self.device,
        )

        self._register_hooks()

    def _calculate_last_filter_outputs(self) -> int:
        # Mirrors the helper in `srr_cbp.py` so the GnT receives the
        # same spatial-flattening factor.
        if hasattr(self.net, "get_plasticity_map"):
            try:
                plasticity_map = self.net.get_plasticity_map()
                for item in plasticity_map:
                    current_layer = item['weight_module']
                    outgoing_module = item['outgoing_module']
                    if isinstance(current_layer, nn.Conv2d) and isinstance(outgoing_module, nn.Linear):
                        num_last_filter_outputs = (
                            outgoing_module.in_features // current_layer.out_channels
                        )
                        return max(1, int(num_last_filter_outputs))
                return 1
            except Exception:
                pass

        layers = cast(Sequence[nn.Module], self.net.layers)
        last_conv_idx = -1
        first_linear_idx = -1
        for i in range(0, len(layers), 2):
            if isinstance(layers[i], nn.Conv2d):
                last_conv_idx = i
            elif isinstance(layers[i], nn.Linear):
                if first_linear_idx == -1:
                    first_linear_idx = i
                break
        if last_conv_idx == -1 or first_linear_idx == -1:
            return 1
        last_conv = cast(nn.Conv2d, layers[last_conv_idx])
        first_linear = cast(nn.Linear, layers[first_linear_idx])
        try:
            num_last_filter_outputs = first_linear.in_features // last_conv.out_channels
            return max(1, int(num_last_filter_outputs))
        except Exception:
            return 1


__all__ = ("SRR_SoftOrtho_CBP_for_FC", "SRR_SoftOrtho_CBP_for_ConvNet")
