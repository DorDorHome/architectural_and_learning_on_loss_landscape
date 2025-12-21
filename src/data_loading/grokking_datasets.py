import torch
from torch.utils.data import Dataset
import numpy as np

class ModularArithmeticDataset(Dataset):
    def __init__(self, train: bool, p: int = 113, seed: int = 42):
        self.train = train
        self.p = p
        
        # Generate all pairs
        pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
        
        # Shuffle deterministically
        g = torch.Generator()
        g.manual_seed(seed)
        shuffled_indices = torch.randperm(len(pairs), generator=g)
        pairs = pairs[shuffled_indices]
        
        # Split
        split_idx = len(pairs) // 2
        if self.train:
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]
            
        # Tokens
        self.plus_token = p
        self.equals_token = p + 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        
        # Input: [x, +, y, =]
        input_tensor = torch.tensor([x, self.plus_token, y, self.equals_token], dtype=torch.long)
        
        # Target: (x + y) mod p
        target_token = (x + y) % self.p
        
        return input_tensor, target_token.long()
