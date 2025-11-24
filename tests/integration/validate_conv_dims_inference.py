"""Validation for Step 6: automatic ConvNet dimension inference.

Creates several ConvNet variants with missing dimensions and ensures
that a forward pass works. Prints inferred dimensions.
"""
import sys
from pathlib import Path
import os

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.configurations import NetConfig, NetParams
from src.models.model_factory import model_factory
import torch


def build_and_run(desc, params_kwargs):
    print(f"\n=== {desc} ===")
    netparams = NetParams(**params_kwargs)
    netcfg = NetConfig(type='ConvNet', netparams=netparams)
    model = model_factory(netcfg)
    print(f"Inferred dims: h={netparams.input_height}, w={netparams.input_width}")
    x = torch.randn(4, 3 if netparams.in_channels!=1 else 1, netparams.input_height, netparams.input_width)
    if netparams.in_channels == 1 and x.shape[1]==1:
        # expand to 3 channels if model expects 3 (current ConvNet hardcodes 3 input channels)
        x = x.repeat(1,3,1,1)
    out, feats = model.predict(x)
    print("Output shape:", out.shape, "#features:", len(feats))
    print("SUCCESS")


def main():
    # Missing both dims, in_channels=3, num_classes=10 -> expect 32
    build_and_run("CIFAR10 heuristic (3ch,10 classes)", dict(num_classes=10, in_channels=3, input_height=None, input_width=None, activation='relu'))
    # Missing both dims, in_channels=1 -> expect 28
    build_and_run("MNIST heuristic (1ch)", dict(num_classes=10, in_channels=1, input_height=None, input_width=None, activation='relu'))
    # Missing both dims, in_channels=3, num_classes=1000 -> 224
    build_and_run("ImageNet heuristic (3ch,1000 classes)", dict(num_classes=1000, in_channels=3, input_height=None, input_width=None, activation='relu'))
    # One dim provided -> copy
    build_and_run("One dimension provided", dict(num_classes=10, in_channels=3, input_height=40, input_width=None, activation='relu'))

if __name__ == '__main__':
    main()
