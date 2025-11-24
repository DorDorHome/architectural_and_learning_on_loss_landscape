"""Quick integration test for RR-CBP with bias transfer."""

import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from configs.configurations import ExperimentConfig
from src.models.model_factory import model_factory
from src.data_loading.dataset_factory import dataset_factory
from src.data_loading.transform_factory import transform_factory
from src.algos.supervised.supervised_factory import create_learner


def test_rr_cbp_integration():
    """Test RR-CBP integration with the training pipeline."""
import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

    
    # Create minimal config for testing
    cfg_dict = {
        'runs': 1,
        'run_id': 0,
        'seed': 42,
        'device': 'cpu',
        'epochs': 2,  # Very short test
        'batch_size': 32,
        'num_tasks': 2,
        'num_workers': 0,
        'use_wandb': False,
        'use_json': False,
        'data': {
            'dataset': 'MNIST',
            'use_torchvision': True,
            'data_path': './data',
            'num_classes': 10
        },
        'net': {
            'type': 'ConvNet',
            'network_class': 'conv',
            'device': 'cpu',
            'netparams': {
                'pretrained': False,
                'num_classes': 10,
                'initialization': 'kaiming',
                'activation': 'leaky_relu',
                'input_height': 28,
                'input_width': 28,
            }
        },
        'learner': {
            'type': 'rr_cbp',
            'network_class': 'conv',
            'init': 'kaiming',
            'device': 'cpu',
            'opt': 'adam',
            'loss': 'cross_entropy',
            'step_size': 0.001,
            'neurons_replacement_rate': 0.1,
            'decay_rate_utility_track': 0.9,
            'maturity_threshold': 5,
            'util_type': 'contribution',
            'accumulate': False,
            'diag_sigma_only': True,
            'sigma_ema_beta': 0.9,
            'sigma_ridge': 1e-2,
            'max_proj_trials': 4,
            'proj_eps': 1e-6,
            'center_bias': 'mean',
            'nullspace_seed_epsilon': 0.0,
            'orthonormalize_batch': True,
            'improve_conditioning_if_saturated': True,
            'log_rank_metrics_every': 1,
        },
        'evaluation': {
            'use_testset': False,
            'eval_freq_epoch': 1,
            'eval_metrics': ['accuracy', 'loss']
        },
        'track_rank': False,  # Disable for speed
        'track_dead_units': False,
        'track_weight_magnitude': False,
    }
    
    cfg = OmegaConf.create(cfg_dict)
    
    # Set up model and data
    transform = transform_factory(cfg.data.dataset, cfg.net.type)
    train_set, _ = dataset_factory(cfg.data, transform=transform, with_testset=False)
    
    cfg.net.netparams.input_height = train_set[0][0].shape[1]
    cfg.net.netparams.input_width = train_set[0][0].shape[2]
    
    net = model_factory(cfg.net)
    net.to(cfg.device)
    
    # Create RR-CBP learner
    learner = create_learner(cfg.learner, net, cfg.net)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    print("🔧 Testing RR-CBP with bias transfer...")
    
    # Run a few training steps
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for epoch in range(cfg.epochs):
        for batch_idx, (input_data, target) in enumerate(train_loader):
            if batch_idx >= 5:  # Only test a few batches
                break
                
            input_data = input_data.to(cfg.device)
            target = target.to(cfg.device)
            
            loss, output = learner.learn(input_data, target)
            
            # Verify finite values
            assert torch.isfinite(loss).item(), f"Loss is not finite: {loss}"
            assert torch.isfinite(output).all(), "Output contains non-finite values"
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(output, 1)
                correct = predicted.eq(target).sum().item()
                accuracy = correct / input_data.size(0)
            
            total_loss += loss.item()
            total_acc += accuracy
            num_batches += 1
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}, Acc={accuracy:.4f}")
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    # Check replacement statistics
    stats = learner.rr_gnt.get_layer_stats() if hasattr(learner, 'rr_gnt') else {}
    total_replacements = sum(s.successful + s.fallbacks for s in stats.values())
    
    print(f"✅ Integration test passed!")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Average Accuracy: {avg_acc:.4f}")
    print(f"   Total Unit Replacements: {total_replacements}")
    print(f"   Network remained stable with bias transfer")
    
    return True


if __name__ == "__main__":
    success = test_rr_cbp_integration()
    if success:
        print("🎉 RR-CBP integration test completed successfully!")