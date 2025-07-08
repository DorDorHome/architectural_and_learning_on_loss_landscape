"""
Research-based investigation of weight normalization instability.
Based on actual findings from the literature, not speculation.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class WeightNormInstabilityInvestigator:
    """
    Investigate the actual causes of weight normalization instability
    based on research literature findings.
    """
    
    def __init__(self, model):
        self.model = model
        self.activation_stats = defaultdict(list)
        self.weight_scale_stats = defaultdict(list)
        self.effective_learning_rates = defaultdict(list)
        self.step_count = 0
        
    def investigate_scale_parameter_explosion(self):
        """
        Huang et al. (2017): Scale parameters can explode due to poor conditioning.
        Check if weight_scale parameters are growing exponentially.
        """
        scale_explosion_report = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_scale'):
                weight_scale = module.weight_scale.detach()
                
                # Check for exponential growth pattern
                if len(self.weight_scale_stats[name]) > 10:
                    recent_scales = [s['max_abs'] for s in self.weight_scale_stats[name][-10:]]
                    growth_rate = recent_scales[-1] / recent_scales[0] if recent_scales[0] != 0 else float('inf')
                    
                    scale_explosion_report[name] = {
                        'current_max_scale': weight_scale.abs().max().item(),
                        'growth_rate_10steps': growth_rate,
                        'potentially_exploding': growth_rate > 2.0,
                        'scale_variance': weight_scale.var().item()
                    }
        
        return scale_explosion_report
    
    def investigate_activation_distribution_mismatch(self):
        """
        Check if ELU's negative mean activations cause distribution mismatch
        that interacts poorly with weight normalization.
        """
        activation_report = {}
        
        for name, module in self.model.named_modules():
            if name in self.activation_stats and len(self.activation_stats[name]) > 0:
                recent_stats = self.activation_stats[name][-5:]  # Last 5 recordings
                
                if recent_stats:
                    means = [s['output_mean'] for s in recent_stats]
                    stds = [s['output_std'] for s in recent_stats]
                    
                    activation_report[name] = {
                        'avg_output_mean': np.mean(means),
                        'avg_output_std': np.mean(stds),
                        'mean_shift_from_zero': abs(np.mean(means)),
                        'std_instability': np.std(stds),
                        'problematic_negative_bias': np.mean(means) < -0.5,
                        'high_variance_instability': np.std(stds) > 1.0
                    }
        
        return activation_report
    
    def investigate_effective_learning_rate_explosion(self):
        """
        Weight normalization can cause effective learning rates to vary wildly.
        w = g * v/||v|| means updates to g scale the entire weight vector.
        """
        lr_report = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_scale'):
                if module.weight.grad is not None and module.weight_scale.grad is not None:
                    
                    # Compute effective learning rate for weight_scale
                    weight_scale_magnitude = module.weight_scale.abs().mean().item()
                    weight_scale_grad_magnitude = module.weight_scale.grad.abs().mean().item()
                    
                    # Effective LR = gradient_magnitude / parameter_magnitude
                    if weight_scale_magnitude > 1e-8:
                        effective_lr_scale = weight_scale_grad_magnitude / weight_scale_magnitude
                    else:
                        effective_lr_scale = float('inf')
                    
                    # Compute effective learning rate for weight direction
                    weight_grad_norm = module.weight.grad.norm().item()
                    weight_norm = module.weight.norm().item()
                    
                    if weight_norm > 1e-8:
                        effective_lr_weight = weight_grad_norm / weight_norm
                    else:
                        effective_lr_weight = float('inf')
                    
                    lr_report[name] = {
                        'effective_lr_scale': effective_lr_scale,
                        'effective_lr_weight': effective_lr_weight,
                        'lr_ratio_scale_to_weight': effective_lr_scale / (effective_lr_weight + 1e-8),
                        'problematic_scale_lr': effective_lr_scale > 1.0,
                        'weight_scale_magnitude': weight_scale_magnitude
                    }
        
        return lr_report
    
    def investigate_initialization_weight_decay_mismatch(self):
        """
        Arpit et al. (2016): Scale parameters need different regularization.
        Check if standard weight decay is insufficient for weight_scale.
        """
        regularization_report = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_scale'):
                
                # Compare weight vs weight_scale magnitudes
                weight_l2 = module.weight.norm().item()
                scale_l2 = module.weight_scale.norm().item()
                
                # Check if weight_scale is growing relative to weight
                scale_to_weight_ratio = scale_l2 / (weight_l2 + 1e-8)
                
                regularization_report[name] = {
                    'weight_l2_norm': weight_l2,
                    'weight_scale_l2_norm': scale_l2,
                    'scale_to_weight_ratio': scale_to_weight_ratio,
                    'scale_dominates_weight': scale_to_weight_ratio > 10.0,
                    'weight_scale_unregularized': scale_l2 > 100.0  # Arbitrary threshold
                }
        
        return regularization_report
    
    def record_training_step(self, inputs_by_layer=None):
        """Record statistics for analysis."""
        self.step_count += 1
        
        # Record weight_scale evolution
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_scale'):
                scale_stats = {
                    'step': self.step_count,
                    'mean': module.weight_scale.mean().item(),
                    'max_abs': module.weight_scale.abs().max().item(),
                    'variance': module.weight_scale.var().item()
                }
                self.weight_scale_stats[name].append(scale_stats)
        
        # Record activation statistics if provided
        if inputs_by_layer:
            for layer_name, (input_tensor, output_tensor) in inputs_by_layer.items():
                if output_tensor is not None:
                    act_stats = {
                        'step': self.step_count,
                        'output_mean': output_tensor.mean().item(),
                        'output_std': output_tensor.std().item(),
                        'output_min': output_tensor.min().item(),
                        'output_max': output_tensor.max().item()
                    }
                    self.activation_stats[layer_name].append(act_stats)
    
    def generate_comprehensive_research_report(self):
        """Generate a research report based on literature findings."""
        report = []
        report.append("=== WEIGHT NORMALIZATION INSTABILITY RESEARCH REPORT ===")
        report.append("Based on findings from:")
        report.append("- Salimans & Kingma (2016): Weight Normalization")
        report.append("- Huang et al. (2017): Centered Weight Normalization") 
        report.append("- Arpit et al. (2016): Regularization of normalized layers")
        report.append("")
        
        # Investigation 1: Scale Parameter Explosion (Huang et al.)
        scale_report = self.investigate_scale_parameter_explosion()
        report.append("1. SCALE PARAMETER EXPLOSION ANALYSIS:")
        exploding_layers = [name for name, data in scale_report.items() 
                           if data.get('potentially_exploding', False)]
        if exploding_layers:
            report.append(f"   ❌ Exploding scale parameters detected in: {exploding_layers}")
            for layer in exploding_layers:
                data = scale_report[layer]
                report.append(f"   - {layer}: growth_rate={data['growth_rate_10steps']:.2f}x, max_scale={data['current_max_scale']:.2e}")
        else:
            report.append("   ✅ No scale parameter explosion detected")
        report.append("")
        
        # Investigation 2: Activation Distribution Issues
        activation_report = self.investigate_activation_distribution_mismatch()
        report.append("2. ACTIVATION DISTRIBUTION ANALYSIS:")
        problematic_activations = [name for name, data in activation_report.items() 
                                 if data.get('problematic_negative_bias', False) or 
                                    data.get('high_variance_instability', False)]
        if problematic_activations:
            report.append(f"   ⚠️  Problematic activation patterns in: {problematic_activations}")
            for layer in problematic_activations:
                data = activation_report[layer]
                report.append(f"   - {layer}: mean={data['avg_output_mean']:.3f}, std_instability={data['std_instability']:.3f}")
        else:
            report.append("   ✅ Activation distributions appear stable")
        report.append("")
        
        # Investigation 3: Effective Learning Rate Issues
        lr_report = self.investigate_effective_learning_rate_explosion()
        report.append("3. EFFECTIVE LEARNING RATE ANALYSIS:")
        problematic_lr = [name for name, data in lr_report.items() 
                         if data.get('problematic_scale_lr', False)]
        if problematic_lr:
            report.append(f"   ❌ Exploding effective learning rates in: {problematic_lr}")
            for layer in problematic_lr:
                data = lr_report[layer]
                report.append(f"   - {layer}: effective_lr_scale={data['effective_lr_scale']:.2e}")
        else:
            report.append("   ✅ Effective learning rates appear controlled")
        report.append("")
        
        # Investigation 4: Regularization Mismatch
        reg_report = self.investigate_initialization_weight_decay_mismatch()
        report.append("4. REGULARIZATION EFFECTIVENESS ANALYSIS:")
        unregularized_scales = [name for name, data in reg_report.items() 
                               if data.get('scale_dominates_weight', False)]
        if unregularized_scales:
            report.append(f"   ❌ Insufficient weight_scale regularization in: {unregularized_scales}")
            for layer in unregularized_scales:
                data = reg_report[layer]
                report.append(f"   - {layer}: scale/weight_ratio={data['scale_to_weight_ratio']:.2f}")
        else:
            report.append("   ✅ Weight decay appears effective on scale parameters")
        report.append("")
        
        # Research recommendations
        report.append("RESEARCH-BASED RECOMMENDATIONS:")
        if exploding_layers:
            report.append("• Scale explosion detected → Try Centered Weight Normalization (Huang et al.)")
            report.append("• Consider separate learning rate for weight_scale parameters")
        if unregularized_scales:
            report.append("• Insufficient regularization → Increase weight_decay or separate scale regularization")
        if problematic_activations:
            report.append("• Activation distribution issues → Consider Group Normalization or LayerNorm")
        if problematic_lr:
            report.append("• Learning rate explosion → Reduce base learning rate or use gradient clipping")
        
        return "\n".join(report)
