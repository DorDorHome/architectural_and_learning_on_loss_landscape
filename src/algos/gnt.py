import sys
import torch
from math import sqrt
import torch.nn.functional as F

from torch.nn import Conv2d, Linear
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
from torch.nn.init import calculate_gain
from src.utils.miscellaneous import get_layer_bound
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.algos.AdamGnT import AdamGnT


class GnT_for_FC(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on maturity-threshold based replacement
    GnT does: 
        (a) compute/decay per-neuron utilities,
        (b) decide which mature low-utility neurons to replace,
        (c) re-initialize their incoming weights (and adjust downstream bias for FC),
        (d) zero related optimizer state so the new weights start “fresh”.

    """
    def __init__(
            self,
            net,
            hidden_activation,
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            util_type='contribution',
            loss_func=F.mse_loss,
            accumulate=False,
    ):
        super(GnT_for_FC, self).__init__()
        self.device = device
        
        
        # self.net = net
        # self.num_hidden_layers = int(len(self.net)/2)
        
        # --- REFACTORED INIT LOGIC ---
        self.plasticity_map = None
        self.use_map = False

        if hasattr(net, 'get_plasticity_map'):
            self.plasticity_map = net.get_plasticity_map()
            self.use_map = True
            self.net = net
            self.num_hidden_layers = len(self.plasticity_map)
        elif hasattr(net, 'layers'):
            self.net = net.layers
            self.num_hidden_layers = int(len(self.net)/2)
        else:
            self.net = net
            self.num_hidden_layers = int(len(self.net)/2)
        # -----------------------------
        
        
        self.loss_func = loss_func
        self.accumulate = accumulate

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'adam'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        
        self.util = []
        self.bias_corrected_util = []
        self.ages = []
        self.mean_feature_act = []
        
        for i in range(self.num_hidden_layers):
            if self.use_map:
                layer = self.plasticity_map[i]['weight_module']
            else:
                layer = self.net[i * 2]
                
           # --- ROBUST DIMENSION LOOKUP ---
            # Handle both Linear (out_features) and Conv2d (out_channels)
            if hasattr(layer, 'out_features'):
                out_feats = layer.out_features
            elif hasattr(layer, 'out_channels'):
                out_feats = layer.out_channels
            else:
                # Fallback or error if using an unsupported layer type provided by map
                raise AttributeError(f"Layer {layer} in plasticity map has neither .out_features nor .out_channels")
            # --------
            self.util.append(torch.zeros(out_feats).to(self.device))
            self.bias_corrected_util.append(torch.zeros(out_feats).to(self.device))
            self.ages.append(torch.zeros(out_feats).to(self.device))
            self.mean_feature_act.append(torch.zeros(out_feats).to(self.device))

        
        # self.util = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        # self.bias_corrected_util = \
        #     [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        # self.ages = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        
        # self.mean_feature_act = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]


        self.m = torch.nn.Softmax(dim=1)
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        
        # --- Use Map or Legacy ---
        if self.use_map:
            bounds = []
            for i in range(self.num_hidden_layers):
                # Map doesn't strictly store indices, but we strictly need the current layer's in_features
                # and hidden_activation logic.
                layer = self.plasticity_map[i]['weight_module']
                if init == 'default':
                    b = sqrt(1 / layer.in_features)
                elif init == 'xavier':
                    b = torch.nn.init.calculate_gain(nonlinearity=hidden_activation) * \
                        sqrt(6 / (layer.in_features + layer.out_features))
                elif init == 'lecun':
                    b = sqrt(3 / layer.in_features)
                else:
                    b = torch.nn.init.calculate_gain(nonlinearity=hidden_activation) * \
                        sqrt(3 / layer.in_features)
                bounds.append(b)
            # Last layer bound (heuristic from original code) usually looks at the "next" layer of the last block
            # For now, we append a dummy bound or replicate the last one to be safe, 
            # as gen_new_features loop index goes up to num_hidden_layers
            # The original code did: bounds.append(1 * sqrt(3 / net[last*2].in_features))
            # We will approximate this using the last layer in map
            last_layer = self.plasticity_map[-1]['weight_module']
            bounds.append(1 * sqrt(3 / last_layer.in_features))
            return bounds
        # -----------------------------------
        # legacy logic starts here
        if init == 'default':
            bounds = [sqrt(1 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        elif init == 'xavier':
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(6 / (self.net[i * 2].in_features + self.net[i * 2].out_features)) for i in
                      range(self.num_hidden_layers)]
        elif init == 'lecun':
            bounds = [sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        else:
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        bounds.append(1 * sqrt(3 / self.net[self.num_hidden_layers * 2].in_features))
        return bounds

    def update_utility(self, layer_idx=0, features=None, next_features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            """
            Adam-style bias correction
            """
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] -= - (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            # --- Use Map or Legacy ---
            if self.use_map:
                map_item = self.plasticity_map[layer_idx]
                current_layer = map_item['weight_module']
                next_layer = map_item['outgoing_module']
            else:
                current_layer = self.net[layer_idx * 2]
                next_layer = self.net[layer_idx * 2 + 2]
            # -----------------------------------
            
            output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
            input_weight_mag = current_layer.weight.data.abs().mean(dim=1)

            if self.util_type == 'weight':
                new_util = output_weight_mag
            elif self.util_type == 'contribution':
                new_util = output_weight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptation':
                new_util = 1/input_weight_mag
            elif self.util_type == 'zero_contribution':
                new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
            elif self.util_type == 'feature_by_input':
                input_weight_mag = self.net[layer_idx*2].weight.data.abs().mean(dim=1)
                new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
            else:
                new_util = 0

            self.util[layer_idx] += (1 - self.decay_rate) * new_util

            """
            Adam-style bias correction
            """
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = torch.rand(self.util[layer_idx].shape)

    def test_features(self, features):
        """
        Select features to replace based on their utility and maturity.
        
        - ages incremented for each layer.
        

        
        
        
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace = [torch.empty(0, dtype=torch.long).to(self.device) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            else:
                if num_new_features_to_replace < 1:
                    if torch.rand(1) <= num_new_features_to_replace:
                        num_new_features_to_replace = 1
                num_new_features_to_replace = int(num_new_features_to_replace)
    
            if num_new_features_to_replace == 0:
                continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = torch.topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                                 num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                
                
                # --- REFACTOR: Map or Legacy ---
                if self.use_map:
                    map_item = self.plasticity_map[i]
                    current_layer = map_item['weight_module']
                    next_layer = map_item['outgoing_module']
                    
                    # CRITICAL FIX: We check if the OUTGOING module feeds into a norm.
                    # If the next layer is normalized, bias compensation is useless/harmful.
                    should_compensate = not map_item.get('outgoing_feeds_into_norm', False)
                else:
                    current_layer = self.net[i * 2]
                    next_layer = self.net[i * 2 + 2]
                    # Legacy fallback: We assume standard FC/Conv blocks DO tolerate compensation
                    should_compensate = True
                # -------------------------------
                
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                # noinspection PyArgumentList
                current_layer.weight.data[features_to_replace[i], :] += \
                    torch.empty(num_features_to_replace[i], current_layer.in_features).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)
                current_layer.bias.data[features_to_replace[i]] *= 0
                """
                # Update bias to correct for the removed features and set the outgoing weights and ages to zero
                """
                if should_compensate:
                    next_layer.bias.data += (next_layer.weight.data[:, features_to_replace[i]] * \
                                                    self.mean_feature_act[i][features_to_replace[i]] / \
                                                    (1 - self.decay_rate ** self.ages[i][features_to_replace[i]])).sum(dim=1)
                
                next_layer.weight.data[:, features_to_replace[i]] = 0
                self.ages[i][features_to_replace[i]] = 0


    def update_optim_params(self, features_to_replace, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'adam':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace[i] == 0:
                    continue
                
                # --- REFACTOR: Map or Legacy ---
                if self.use_map:
                    map_item = self.plasticity_map[i]
                    curr_weight = map_item['weight_module'].weight
                    curr_bias = map_item['weight_module'].bias
                    next_weight = map_item['outgoing_module'].weight
                else:
                    curr_weight = self.net[i * 2].weight
                    curr_bias = self.net[i * 2].bias
                    next_weight = self.net[i * 2 + 2].weight
                # -------------------------------

                self.opt.state[curr_weight]['exp_avg'][features_to_replace[i], :] = 0.0
                self.opt.state[curr_bias]['exp_avg'][features_to_replace[i]] = 0.0
                self.opt.state[curr_weight]['exp_avg_sq'][features_to_replace[i], :] = 0.0
                self.opt.state[curr_bias]['exp_avg_sq'][features_to_replace[i]] = 0.0
                self.opt.state[curr_weight]['step'][features_to_replace[i], :] = 0
                self.opt.state[curr_bias]['step'][features_to_replace[i]] = 0
                # output weights
                self.opt.state[next_weight]['exp_avg'][:, features_to_replace[i]] = 0.0
                self.opt.state[next_weight]['exp_avg_sq'][:, features_to_replace[i]] = 0.0
                self.opt.state[next_weight]['step'][:, features_to_replace[i]] = 0

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)


class ConvGnT_for_ConvNet(object):
    """
    Generate-and-Test algorithm for ConvNets, maturity threshold based tester, accumulates probability of replacement,
    with various measures of feature utility
    """
    def __init__(self, net,
                 hidden_activation,
                 opt,  replacement_rate=1e-4,
                 decay_rate=0.99,init='kaiming',
                 num_last_filter_outputs=4, util_type='contribution',
                 maturity_threshold=100, device='cpu'):
        super(ConvGnT_for_ConvNet, self).__init__()
        self.plasticity_map = None
        self.use_map = False

        if hasattr(net, 'get_plasticity_map'):
            # Case 1: Advanced Model with Explicit Map
            self.plasticity_map = net.get_plasticity_map()
            self.use_map = True
            self.net = net
            self.num_hidden_layers = len(self.plasticity_map)
        elif hasattr(net, 'layers'):
            # Case 2: Standard Model passed as Object (Robustness/Transition)
            # Unwrap the layers so legacy logic works
            self.net = net.layers
            self.num_hidden_layers = int(len(self.net)/2)
        else:
            # Case 3: Legacy Usage (Passing list/Sequential directly)
            self.net = net
            self.num_hidden_layers = int(len(self.net)/2)
        
        

        # self.net = net
        # self.num_hidden_layers = int(len(self.net)/2)
        self.util_type = util_type
        self.device = device

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'AdamGnT'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.num_last_filter_outputs = num_last_filter_outputs
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util, self.bias_corrected_util, self.ages, self.mean_feature_act, self.mean_abs_feature_act, \
             = [], [], [], [], []

        for i in range(self.num_hidden_layers):
            if self.use_map:
                current_layer = self.plasticity_map[i]['weight_module']
            else:
                current_layer = self.net[i * 2]
                
            # Robust dimension check
            if hasattr(current_layer, 'out_channels'):
                out_feats = current_layer.out_channels
            else:
                out_feats = current_layer.out_features
                
            self.util.append(zeros(out_feats).to(self.device))
            self.bias_corrected_util.append(zeros(out_feats).to(self.device))
            self.ages.append(zeros(out_feats).to(self.device))
            self.mean_feature_act.append(zeros(out_feats).to(self.device))
            self.mean_abs_feature_act.append(zeros(out_feats).to(self.device))

            
            # if isinstance(self.net[i * 2], Conv2d):
            #     self.util.append(zeros(self.net[i * 2].out_channels).to(self.device))
            #     self.bias_corrected_util.append(zeros(self.net[i * 2].out_channels).to(self.device))
            #     self.ages.append(zeros(self.net[i * 2].out_channels).to(self.device))
            #     self.mean_feature_act.append(zeros(self.net[i * 2].out_channels).to(self.device))
            #     self.mean_abs_feature_act.append(zeros(self.net[i * 2].out_channels).to(self.device))
            # elif isinstance(self.net[i * 2], Linear):
            #     self.util.append(zeros(self.net[i * 2].out_features).to(self.device))
            #     self.bias_corrected_util.append(zeros(self.net[i * 2].out_features).to(self.device))
            #     self.ages.append(zeros(self.net[i * 2].out_features).to(self.device))
            #     self.mean_feature_act.append(zeros(self.net[i * 2].out_features).to(self.device))
            #     self.mean_abs_feature_act.append(zeros(self.net[i * 2].out_features).to(self.device))

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)
        """
        Pre calculate number of features to replace per layer per update
        """
        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                if self.use_map:
                    layer = self.plasticity_map[i]['weight_module']
                else:
                    layer = self.net[i * 2]

                if isinstance(layer, Linear):
                    self.num_new_features_to_replace.append(self.replacement_rate * layer.out_features)
                elif isinstance(layer, Conv2d):
                    self.num_new_features_to_replace.append(self.replacement_rate * layer.out_channels)

                # if isinstance(self.net[i * 2], Linear):
                #     self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_features)
                # elif isinstance(self.net[i * 2], Conv2d):
                #     self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_channels)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        if self.use_map:
            bounds = []
            gain = calculate_gain(nonlinearity=hidden_activation)
            for i in range(self.num_hidden_layers):
                layer = self.plasticity_map[i]['weight_module']
                bounds.append(get_layer_bound(layer=layer, init=init, gain=gain))
            # Heuristic for output layer
            bounds.append(get_layer_bound(layer=self.plasticity_map[-1]['weight_module'], init=init, gain=1))
            return bounds
        
        
        bounds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            bounds.append(get_layer_bound(layer=self.net[i * 2], init=init, gain=gain))
        bounds.append(get_layer_bound(layer=self.net[-1], init=init, gain=1))
        return bounds

    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            # --- REFACTOR: Use Map or Legacy ---
            if self.use_map:
                map_item = self.plasticity_map[layer_idx]
                current_layer = map_item['weight_module']
                next_layer = map_item['outgoing_module']
            else:
                current_layer = self.net[layer_idx * 2]
                next_layer = self.net[layer_idx * 2 + 2]
            # -----------------------------------

            # current_layer = self.net[layer_idx * 2]
            # next_layer = self.net[layer_idx * 2 + 2]

            if isinstance(next_layer, Linear):
                output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
            elif isinstance(next_layer, Conv2d):
                output_weight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_abs_feature_act[layer_idx] *= self.decay_rate
            if isinstance(current_layer, Linear):
                input_weight_mag = current_layer.weight.data.abs().mean(dim=1)
                self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=0)
                self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=0)
            elif isinstance(current_layer, Conv2d):
                input_weight_mag = current_layer.weight.data.abs().mean(dim=(1, 2, 3))
                if isinstance(next_layer, Conv2d):
                    self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=(0, 2, 3))
                    self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=(0, 2, 3))
                else:
                    self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)
                    self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)

            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            if self.util_type == 'adaptation':
                new_util = 1 / input_weight_mag
            elif self.util_type in ['contribution', 'zero_contribution', 'adaptable_contribution']:
                if self.util_type == 'contribution':
                    bias_corrected_act = 0
                else:
                    if isinstance(current_layer, Conv2d):
                        if isinstance(next_layer, Conv2d):
                            bias_corrected_act = bias_corrected_act.view(1, -1, 1, 1)
                        else:
                            bias_corrected_act = bias_corrected_act.repeat_interleave(self.num_last_filter_outputs).view(1, -1)
                if isinstance(next_layer, Linear):
                    if isinstance(current_layer, Linear):
                        new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)
                    elif isinstance(current_layer, Conv2d):
                        new_util = (output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)).view(-1, self.num_last_filter_outputs).mean(dim=1)
                elif isinstance(next_layer, Conv2d):
                    new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=(0, 2, 3))
                if self.util_type == 'adaptable_contribution':
                    new_util = new_util / input_weight_mag

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = rand(self.util[layer_idx].shape)
            else:
                self.util[layer_idx] += (1 - self.decay_rate) * new_util
                # correct the bias in the utility computation
                self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace_input_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        features_to_replace_output_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[i] += self.num_new_features_to_replace[i]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0:    continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                           num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.
            self.mean_abs_feature_act[i][new_features_to_replace] = 0.

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace_input_indices[i] = new_features_to_replace
            features_to_replace_output_indices[i] = new_features_to_replace
            
            # --- REFACTOR: Use Map or Legacy ---
            if self.use_map:
                map_item = self.plasticity_map[i]
                current_layer = map_item['weight_module']
                next_layer = map_item['outgoing_module']
            else:
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]
            # --
            
            # if isinstance(self.net[i * 2], Conv2d) and isinstance(self.net[i * 2 + 2], Linear):
            if isinstance(current_layer, Conv2d) and isinstance(next_layer, Linear):
                features_to_replace_output_indices[i] = \
                    (new_features_to_replace*self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) + \
                    tensor([i for i in range(self.num_last_filter_outputs)]).repeat(new_features_to_replace.size()[0]).to(self.device)

        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

    def update_optim_params(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'AdamGnT':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace[i] == 0:
                    continue
                # --- REFACTOR: Use Map or Legacy ---
                if self.use_map:
                    map_item = self.plasticity_map[i]
                    curr_bias = map_item['weight_module'].bias
                    curr_weight = map_item['weight_module'].weight
                    next_weight = map_item['outgoing_module'].weight
                else:
                    curr_bias = self.net[i * 2].bias
                    curr_weight = self.net[i * 2].weight
                    next_weight = self.net[i * 2 + 2].weight
                # -----------------------------------
                # input weights
                self.opt.state[curr_bias]['exp_avg'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[curr_weight]['exp_avg_sq'][features_to_replace_input_indices[i], :] = 0.0
                self.opt.state[curr_bias]['exp_avg_sq'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[curr_weight]['step'][features_to_replace_input_indices[i], :] = 0
                self.opt.state[curr_bias]['step'][features_to_replace_input_indices[i]] = 0
                # output weights
                self.opt.state[next_weight]['exp_avg'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[next_weight]['exp_avg_sq'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[next_weight]['step'][:, features_to_replace_output_indices[i]] = 0

                
                
                # input weights
                # self.opt.state[self.net[i * 2].bias]['exp_avg'][features_to_replace_input_indices[i]] = 0.0
                # self.opt.state[self.net[i * 2].weight]['exp_avg_sq'][features_to_replace_input_indices[i], :] = 0.0
                # self.opt.state[self.net[i * 2].bias]['exp_avg_sq'][features_to_replace_input_indices[i]] = 0.0
                # self.opt.state[self.net[i * 2].weight]['step'][features_to_replace_input_indices[i], :] = 0
                # self.opt.state[self.net[i * 2].bias]['step'][features_to_replace_input_indices[i]] = 0
                # # output weights
                # self.opt.state[self.net[i * 2 + 2].weight]['exp_avg'][:, features_to_replace_output_indices[i]] = 0.0
                # self.opt.state[self.net[i * 2 + 2].weight]['exp_avg_sq'][:, features_to_replace_output_indices[i]] = 0.0
                # self.opt.state[self.net[i * 2 + 2].weight]['step'][:, features_to_replace_output_indices[i]] = 0

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                # --- REFACTOR: Map or Legacy ---
                if self.use_map:
                    map_item = self.plasticity_map[i]
                    current_layer = map_item['weight_module']
                    next_layer = map_item['outgoing_module']
                    should_compensate = not map_item.get('outgoing_feeds_into_norm', False)
                else:
                    current_layer = self.net[i * 2]
                    next_layer = self.net[i * 2 + 2]
                    should_compensate = True
                # -------------------------------

                if isinstance(current_layer, Linear):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty(num_features_to_replace[i], current_layer.in_features).uniform_(-self.bounds[i],
                                                                                                self.bounds[i]).to(self.device)
                elif isinstance(current_layer, Conv2d):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:])). \
                            uniform_(-self.bounds[i], self.bounds[i]).to(self.device)

                current_layer.bias.data[features_to_replace_input_indices[i]] *= 0.0
                """
                # Update bias to correct for the removed features
                """
                if should_compensate:
                    next_layer.bias.data += (next_layer.weight.data[:, features_to_replace_output_indices[i]] * \
                                                    self.mean_feature_act[i][features_to_replace_input_indices[i]] / \
                                                    (1 - self.decay_rate ** self.ages[i][features_to_replace_input_indices[i]])).sum(dim=1)
                  
                
                """
                # Set the outgoing weights and ages to zero
                """
                next_layer.weight.data[:, features_to_replace_output_indices[i]] = 0
                self.ages[i][features_to_replace_input_indices[i]] = 0

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
        self.update_optim_params(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
