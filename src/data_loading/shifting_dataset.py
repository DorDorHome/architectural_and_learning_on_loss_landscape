# This file contains classes useful for shifting the dataset.

import torch
from torch.utils.data import Dataset

class PermutedDataset(Dataset):
    def __init__(self, original_dataset, permutation=None, flatten=False, transform=None):
        """
        A dataset wrapper that applies a fixed pixel permutation per task.

        Args:
            original_dataset (Dataset): The base dataset (e.g., MNIST, CIFAR10).
            permutation (torch.Tensor or list, optional): Indices for pixel shuffling.
                - If flatten=True: Size must match C*H*W (flattened image size).
                - If flatten=False: Size must match H*W (spatial grid size).
                If None, no permutation is applied.
            flatten (bool): If True, flattens images into 1D vectors after processing.
            transform (callable, optional): Transform to apply after permutation.
        """
        self.original_dataset = original_dataset
        self.flatten = flatten
        self.transform = transform

        # Infer image shape from the first sample
        sample_img, _ = self.original_dataset[0]
        if not isinstance(sample_img, torch.Tensor):
            sample_img = torch.tensor(sample_img, dtype=torch.float32)
        self.original_shape = sample_img.shape

        # Determine channels (C), height (H), and width (W)
        if len(self.original_shape) == 3:  # [C, H, W]
            self.C, self.H, self.W = self.original_shape
        elif len(self.original_shape) == 2:  # [H, W]
            self.C, self.H, self.W = 1, self.original_shape[0], self.original_shape[1]
        else:
            raise ValueError(f"Unsupported image shape: {self.original_shape}")

        # Validate and set permutation
        self.permutation = None
        if permutation is not None:
            permutation = torch.tensor(permutation, dtype=torch.long)
            expected_size = (self.C * self.H * self.W) if flatten else (self.H * self.W)
            if len(permutation) != expected_size:
                raise ValueError(f"Permutation size {len(permutation)} must match {expected_size}")
            self.permutation = permutation

    def __getitem__(self, index):
        """Retrieve and process an item from the dataset."""
        img, label = self.original_dataset[index]

        # Convert to tensor and ensure channel dimension
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        if len(img.shape) == 2:  # [H, W] -> [1, H, W]
            img = img.unsqueeze(0)

        # Apply permutation
        if self.permutation is not None:
            if self.flatten:
                # Flatten first, then permute the 1D vector
                img = img.flatten()
                img = img[self.permutation]
            else:
                # Permute spatial grid (H*W) while preserving channels
                C, H, W = img.shape
                flat_img = img.permute(1, 2, 0).reshape(H * W, C)  # [H*W, C]
                permuted_img = flat_img[self.permutation]  # Apply permutation
                img = permuted_img.reshape(H, W, C).permute(2, 0, 1)  # [C, H, W]

        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)

        # Flatten if requested (after transform)
        if self.flatten:
            img = img.flatten()

        return img, label

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.original_dataset)

    def get_input_shape(self):
        """Return the shape of processed data."""
        if self.flatten:
            return (self.C * self.H * self.W,)
        return (self.C, self.H, self.W)
    
if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np

    # Load MNIST dataset
    mnist_train = datasets.MNIST(root="data", train=True, download=False, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transforms.ToTensor())

    # Define a permutation (e.g., random shuffle)
    np.random.seed(0)
    permutation = np.random.permutation(28 * 28)  # Shuffle pixels

    # Wrap MNIST dataset with permutation
    permuted_mnist_train = PermutedDataset(mnist_train, permutation=permutation, flatten=True)
    permuted_mnist_test = PermutedDataset(mnist_test, permutation=permutation, flatten=True)

    # Access an item
    img, label = permuted_mnist_train[0]
    print(img.shape, label)

    # Save an image for visualization
    img = img.reshape(28, 28).numpy()  # Reshape to 2D and convert to NumPy
    img = (img * 255).astype(np.uint8)  # Scale to 0-255
    img = Image.fromarray(img)  # Create an image
    img.save("permuted_mnist_example.png")

    # Check input shape
    print(permuted_mnist_train.get_input_shape())  # Expected: (784,)
    print(permuted_mnist_test.get_input_shape())  # Expected: (784,)
    
    # Output:
    # torch.Size([784]) 5
    # (784,)
    # (784,)