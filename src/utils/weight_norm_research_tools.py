"""
Research-oriented diagnostics for weight normalization instability.
This module provides tools to investigate WHY weight_scale explosion occurs
without artificially constraining the model capacity.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class WeightNormalizationInvestigator:
    """
    A tool to track and analyze weight normalization dynamics during training
    to understand the root causes of instability.
    """
    
    def __init__(self, model, track_frequency=10):
        self.model = model
        self.track_frequency = track_frequency
        self.step_count = 0
        
        # Storage for tracking
        self.weight_scale_history = defaultdict(list)
        self.weight_norm_history = defaultdict(list)
        self.gradient_norm_history = defaultdict(list)
        self.activation_stats_history = defaultdict(list)
        self.weight_scale_gradient_history = defaultdict(list)
        
        # Hooks storage
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to track activations and gradients during training."""
        for name, module in self.model.named_modules():
            if self._is_normalized_layer(module):
                # Forward hook for activations
                def make_forward_hook(layer_name):
                    def forward_hook(module, input, output):
                        if self.step_count % self.track_frequency == 0:
                            self._record_activation_stats(layer_name, input[0], output)
                    return forward_hook
                
                # Backward hook for gradients
                def make_backward_hook(layer_name):
                    def backward_hook(module, grad_input, grad_output):
                        if self.step_count % self.track_frequency == 0:
                            self._record_gradient_stats(layer_name, module, grad_output[0])
                    return backward_hook
                
                forward_handle = module.register_forward_hook(make_forward_hook(name))
                backward_handle = module.register_backward_hook(make_backward_hook(name))
                self.hooks.extend([forward_handle, backward_handle])
    
    def _is_normalized_layer(self, module):
        """Check if module is a normalized weight layer."""
        return any(norm_type in type(module).__name__ 
                  for norm_type in ['Normalized', 'normalized', 'Norm', 'norm'])
    
    def _record_activation_stats(self, layer_name, input_tensor, output_tensor):
        """Record statistics about activations."""
        with torch.no_grad():
            if torch.is_tensor(input_tensor):
                input_stats = {
                    'mean': input_tensor.mean().item(),
                    'std': input_tensor.std().item(),
                    'max_abs': input_tensor.abs().max().item(),
                    'min': input_tensor.min().item(),
                    'max': input_tensor.max().item()
                }
            else:
                input_stats = {'error': 'not_tensor'}
            
            if torch.is_tensor(output_tensor):
                output_stats = {
                    'mean': output_tensor.mean().item(),
                    'std': output_tensor.std().item(),
                    'max_abs': output_tensor.abs().max().item(),
                    'min': output_tensor.min().item(),
                    'max': output_tensor.max().item()
                }
            else:
                output_stats = {'error': 'not_tensor'}
            
            self.activation_stats_history[layer_name].append({
                'step': self.step_count,
                'input': input_stats,
                'output': output_stats
            })
    
    def _record_gradient_stats(self, layer_name, module, grad_output):
        """Record gradient statistics."""
        with torch.no_grad():
            if hasattr(module, 'weight_scale') and module.weight_scale.grad is not None:
                ws_grad_norm = module.weight_scale.grad.norm().item()
                self.weight_scale_gradient_history[layer_name].append({
                    'step': self.step_count,
                    'weight_scale_grad_norm': ws_grad_norm
                })
    
    def record_step(self):
        """Call this after each training step to record current state."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if self._is_normalized_layer(module):
                    
                    # Record weight_scale evolution
                    if hasattr(module, 'weight_scale'):
                        ws_stats = {
                            'step': self.step_count,
                            'mean': module.weight_scale.mean().item(),
                            'std': module.weight_scale.std().item(),
                            'max_abs': module.weight_scale.abs().max().item(),
                            'min': module.weight_scale.min().item(),
                            'max': module.weight_scale.max().item()
                        }
                        self.weight_scale_history[name].append(ws_stats)
                    
                    # Record weight norm evolution
                    if hasattr(module, 'weight'):
                        weight_norms = torch.norm(module.weight, dim=1)
                        wn_stats = {
                            'step': self.step_count,
                            'mean': weight_norms.mean().item(),
                            'std': weight_norms.std().item(),
                            'max': weight_norms.max().item(),
                            'min': weight_norms.min().item()
                        }
                        self.weight_norm_history[name].append(wn_stats)
                    
                    # Record gradient norms
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        grad_norm = module.weight.grad.norm().item()
                        self.gradient_norm_history[name].append({
                            'step': self.step_count,
                            'weight_grad_norm': grad_norm
                        })
        
        self.step_count += 1
    
    def analyze_instability_patterns(self):
        """Analyze the collected data to identify instability patterns."""
        analysis = {}
        
        for layer_name in self.weight_scale_history:
            layer_analysis = {}
            
            # Weight scale evolution
            ws_data = self.weight_scale_history[layer_name]
            if len(ws_data) > 10:
                max_abs_values = [d['max_abs'] for d in ws_data]
                
                # Check for exponential growth
                if len(max_abs_values) > 5:
                    recent_growth = max_abs_values[-1] / max_abs_values[-5]
                    layer_analysis['weight_scale_growth_rate'] = recent_growth
                    layer_analysis['weight_scale_final'] = max_abs_values[-1]
                    layer_analysis['weight_scale_explosive'] = recent_growth > 10.0
                
                # Check for correlation with gradients
                if layer_name in self.weight_scale_gradient_history:
                    grad_data = self.weight_scale_gradient_history[layer_name]
                    if len(grad_data) > 5:
                        recent_grads = [d['weight_scale_grad_norm'] for d in grad_data[-5:]]
                        layer_analysis['avg_recent_weight_scale_grad'] = np.mean(recent_grads)
                        layer_analysis['large_gradients'] = any(g > 1.0 for g in recent_grads)
            
            # Activation explosion patterns
            if layer_name in self.activation_stats_history:
                act_data = self.activation_stats_history[layer_name]
                if len(act_data) > 5:
                    recent_outputs = [d['output']['max_abs'] for d in act_data[-5:] 
                                    if 'error' not in d['output']]
                    if recent_outputs:
                        layer_analysis['avg_output_magnitude'] = np.mean(recent_outputs)
                        layer_analysis['output_explosion'] = any(out > 1e6 for out in recent_outputs)
            
            analysis[layer_name] = layer_analysis
        
        return analysis
    
    def generate_research_report(self, save_path=None):
        """Generate a comprehensive research report on the instability."""
        analysis = self.analyze_instability_patterns()
        
        report = ["=== WEIGHT NORMALIZATION INSTABILITY RESEARCH REPORT ===\n"]
        
        # Summary of findings
        explosive_layers = [name for name, data in analysis.items() 
                           if data.get('weight_scale_explosive', False)]
        
        if explosive_layers:
            report.append(f"🔥 EXPLOSIVE LAYERS DETECTED: {explosive_layers}\n")
        
        # Detailed analysis per layer
        for layer_name, data in analysis.items():
            report.append(f"\n--- LAYER: {layer_name} ---")
            
            if 'weight_scale_growth_rate' in data:
                report.append(f"Weight Scale Growth Rate (last 5 steps): {data['weight_scale_growth_rate']:.2f}x")
                report.append(f"Final Weight Scale Magnitude: {data['weight_scale_final']:.6e}")
            
            if 'avg_recent_weight_scale_grad' in data:
                report.append(f"Average Recent Weight Scale Gradient: {data['avg_recent_weight_scale_grad']:.6e}")
                report.append(f"Large Gradients Detected: {data.get('large_gradients', False)}")
            
            if 'avg_output_magnitude' in data:
                report.append(f"Average Output Magnitude: {data['avg_output_magnitude']:.6e}")
                report.append(f"Output Explosion: {data.get('output_explosion', False)}")
        
        # Research hypotheses
        report.append("\n\n=== RESEARCH HYPOTHESES ===")
        report.append("Based on the data, potential causes of instability:")
        
        has_explosive_growth = any(data.get('weight_scale_explosive', False) for data in analysis.values())
        has_large_gradients = any(data.get('large_gradients', False) for data in analysis.values())
        has_output_explosion = any(data.get('output_explosion', False) for data in analysis.values())
        
        if has_explosive_growth and has_large_gradients:
            report.append("• GRADIENT EXPLOSION: Large gradients causing weight_scale to grow exponentially")
        
        if has_output_explosion:
            report.append("• ACTIVATION EXPLOSION: Output magnitudes suggest forward pass instability")
        
        if has_explosive_growth and not has_large_gradients:
            report.append("• OPTIMIZATION PATHOLOGY: Weight scales growing without correspondingly large gradients")
        
        report.append("\nRecommended investigations:")
        report.append("• Learning rate sensitivity analysis")
        report.append("• Weight decay effectiveness on weight_scale parameters")
        report.append("• Activation function choice impact")
        report.append("• Initialization sensitivity")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Integration with your training loop
class ResearchEnhancedBackprop:
    """
    Enhanced version of your Backprop class that includes research diagnostics
    without compromising model capacity.
    """
    
    def __init__(self, net, config, netconfig=None):
        # Initialize your existing Backprop
        from src.algos.supervised.basic_backprop import Backprop
        self.base_learner = Backprop(net, config, netconfig)
        
        # Add research tools
        self.investigator = WeightNormalizationInvestigator(net, track_frequency=10)
        self.step_count = 0
        
        # Expose base learner attributes
        for attr in ['net', 'device', 'opt', 'loss_func', 'previous_features', 'to_perturb', 'perturb_scale']:
            setattr(self, attr, getattr(self.base_learner, attr))
    
    def learn(self, x, target):
        """Enhanced learn method with research diagnostics."""
        # Call original learn method
        loss, output = self.base_learner.learn(x, target)
        
        # Record research data
        self.investigator.record_step()
        self.step_count += 1
        
        # Generate research report every 100 steps
        if self.step_count % 100 == 0:
            analysis = self.investigator.analyze_instability_patterns()
            
            # Check for early warning signs
            for layer_name, data in analysis.items():
                if data.get('weight_scale_growth_rate', 1.0) > 5.0:
                    print(f"⚠️  Research Alert: {layer_name} showing rapid weight_scale growth!")
                
                if data.get('avg_recent_weight_scale_grad', 0.0) > 1.0:
                    print(f"⚠️  Research Alert: {layer_name} has large weight_scale gradients!")
        
        return loss, output
    
    def generate_final_report(self, save_path):
        """Generate final research report."""
        return self.investigator.generate_research_report(save_path)
    
    def cleanup(self):
        """Cleanup research tools."""
        self.investigator.cleanup()
