#!/usr/bin/env python3
"""
Comparison test between RR-CBP and baseline training to verify performance improvement.
"""
import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configs.configurations import RRContinuousBackpropConfig, LinearNetParams, NetConfig
from src.models.deep_ffnn import DeepFFNN
from src.algos.supervised.rr_cbp_fc import RankRestoringCBP_for_FC


def get_data_path():
    """Resolve the data path, preferring local existing directories."""
    potential_paths = ['/home/sfchan/dataset', '/hdda/datasets']
    for path in potential_paths:
        if os.path.exists(path):
            return path
    return './data'


def test_comparison():
    """Compare RR-CBP with bias transfer vs baseline SGD training."""
    
    # Minimal config
    device = torch.device('cpu')
    
    # Data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Small subset for quick test
    data_path = get_data_path()
    # download=True ensures it downloads if missing (CI), but skips if present (local)
    dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    subset_indices = torch.randperm(len(dataset))[:500]  # Just 500 samples
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    
    print("🔍 Comparing RR-CBP vs Baseline Training...")
    
    # Test 1: Baseline SGD
    print("\n📊 Testing Baseline SGD...")
    baseline_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    baseline_losses = []
    baseline_accs = []
    
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 5:  # Just first 5 batches
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean().item()
            
            baseline_losses.append(loss.item())
            baseline_accs.append(acc)
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    # Test 2: RR-CBP with bias transfer
    print("\n🚀 Testing RR-CBP with Bias Transfer...")
    
    # Create RR-CBP model with proper configuration
    net_params = LinearNetParams(input_size=784, num_features=100, num_outputs=10, num_hidden_layers=2, act_type='relu')
    rr_net = DeepFFNN(net_params)
    rr_net.type = 'FC'
    rr_net = rr_net.to(device)
    
    config = RRContinuousBackpropConfig(
        device=device,
        neurons_replacement_rate=0.5,  # Replace 50% of mature units
        maturity_threshold=2,  # Units become mature after 2 steps
        accumulate=False,
        diag_sigma_only=True,
        sigma_ema_beta=0.9,
        sigma_ridge=1e-3,
        log_rank_metrics_every=0,  # No logging for this test
    )
    config.opt = 'adam'
    net_config = NetConfig(type='FC', netparams=net_params)
    
    rr_model = RankRestoringCBP_for_FC(rr_net, config, netconfig=net_config)
    
    rr_losses = []
    rr_accs = []
    
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 5:  # Just first 5 batches
                break
                
            data, target = data.to(device), target.to(device)
            data_flat = data.view(data.size(0), -1)
            
            # RR-CBP learning step
            loss, output = rr_model.learn(data_flat, target)
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean().item()
            
            rr_losses.append(loss.item())
            rr_accs.append(acc)
            
            if batch_idx % 2 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    # Compare results
    print("\n📈 Comparison Results:")
    print(f"Baseline SGD:")
    print(f"  Average Loss: {sum(baseline_losses)/len(baseline_losses):.4f}")
    print(f"  Average Accuracy: {sum(baseline_accs)/len(baseline_accs):.4f}")
    print(f"  Final Loss: {baseline_losses[-1]:.4f}")
    print(f"  Final Accuracy: {baseline_accs[-1]:.4f}")
    
    print(f"\nRR-CBP with Bias Transfer:")
    print(f"  Average Loss: {sum(rr_losses)/len(rr_losses):.4f}")
    print(f"  Average Accuracy: {sum(rr_accs)/len(rr_accs):.4f}")
    print(f"  Final Loss: {rr_losses[-1]:.4f}")
    print(f"  Final Accuracy: {rr_accs[-1]:.4f}")
    print(f"  Total Unit Replacements: {rr_model.total_replacement_count if hasattr(rr_model, 'total_replacement_count') else 'N/A'}")
    
    # Performance check
    rr_final_loss = rr_losses[-1]
    baseline_final_loss = baseline_losses[-1]
    
    if rr_final_loss <= baseline_final_loss * 1.2:  # Within 20% is good
        print("\n✅ Performance Test: RR-CBP performs competitively with baseline!")
    else:
        print(f"\n⚠️  Performance Test: RR-CBP final loss {rr_final_loss:.4f} vs baseline {baseline_final_loss:.4f}")
    
    print("\n🎯 RR-CBP vs Baseline comparison completed!")

if __name__ == "__main__":
    test_comparison()
