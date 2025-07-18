# This file contains classes useful for shifting the dataset.

import sys
import pathlib
from typing import Any, Callable, Optional, Tuple, Union
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
# import dataset_factory:

from src.data_loading.dataset_factory import dataset_factory


import torch
from torch.utils.data import Dataset
import numpy as np


class Permuted_input_Dataset(Dataset[Tuple[torch.Tensor, Any]]):
    def __init__(self, original_dataset: Dataset, permutation: Optional[Union[np.ndarray, list]] = None, flatten: bool = False, transform: Optional[Callable] = None):
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

        # Determine channels (c), height (h), and width (w)
        if len(self.original_shape) == 3:  # [C, H, W]
            self.c, self.h, self.w = self.original_shape
        elif len(self.original_shape) == 2:  # [H, W]
            self.c, self.h, self.w = 1, self.original_shape[0], self.original_shape[1]
        else:
            raise ValueError(f"Unsupported image shape: {self.original_shape}")

        # Validate and set permutation
        self.permutation = None
        if permutation is not None:
            permutation = torch.tensor(permutation, dtype=torch.long)
            expected_size = (self.c * self.h * self.w) if flatten else (self.h * self.w)
            if len(permutation) != expected_size:
                raise ValueError(f"Permutation size {len(permutation)} must match {expected_size}")
            self.permutation = permutation

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
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

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.original_dataset)

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the shape of processed data."""
        if self.flatten:
            return (self.c * self.h * self.w,)
        return (self.c, self.h, self.w)


class Permuted_output_Dataset(Dataset[Tuple[Any, torch.Tensor]]):
    def __init__(self, original_dataset: Dataset, permutation: Union[np.ndarray, list, torch.Tensor], flatten: bool = False, transform: Optional[Callable] = None):
        """
        A dataset wrapper that applies a fixed permutation to the output labels.

        Args:
            original_dataset (Dataset): The base dataset (e.g., MNIST, CIFAR10).
            permutation (torch.Tensor or list): A tensor or list that maps original
                class indices to new ones. For a dataset with 10 classes, this
                should be a permutation of the integers 0-9.
            flatten (bool): This argument is ignored but included for API consistency.
            transform (callable, optional): This argument is ignored but included for
                API consistency.
        """
        self.original_dataset = original_dataset
        
        if permutation is None:
            raise ValueError("A permutation must be provided for Permuted_output_Dataset.")
            
        self.permutation = torch.tensor(permutation, dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """Retrieve an item and permute its label."""
        img, label = self.original_dataset[index]
        
        # Apply the permutation to the label
        permuted_label = self.permutation[label]
        
        return img, permuted_label

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.original_dataset)

    
if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np
    from configs.configurations import DataConfig

    #set DataConfig attributes:
    data_config = DataConfig()
    data_config.dataset = 'MNIST'
    data_config.use_torchvision = True
    data_config.data_path = '/hdda/datasets'
    data_config.num_classes = 10
    data_config.shuffle = False
    data_config.transform = None
    
    
    
    test_dataconfig2 = DataConfig(dataset='CIFAR10',
                                data_path= "/hdda/datasets",
                                use_torchvision=True)

    
    
    cifar_train, cifar_test = dataset_factory(test_dataconfig2, transform=None)

    
    # use dataset_factory to load the dataset:
    mnist_train, mnist_test = dataset_factory(data_config, transform=transforms.ToTensor(), with_testset=True) 
    
    # print the shape of the dataset:
    print(mnist_train[0][0].shape)  # Example shape of the first image
    
    
    
    


    # Define a permutation (e.g., random shuffle)
    np.random.seed(0)
    permutation = np.random.permutation(28 * 28) if data_config.dataset == 'MNIST' else np.random.permutation(32 * 32 * 3)  # For CIFAR10, use 32x32x3= np.random.permutation(28 * 28)  # Shuffle pixels

    # Wrap MNIST dataset with permutation
    permuted_mnist_train = Permuted_input_Dataset(mnist_train, permutation=permutation, flatten=True)
    permuted_mnist_test = Permuted_input_Dataset(mnist_test, permutation=permutation, flatten=True)

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
    
    # --- Test for Permuted_output_Dataset ---
    print("\n--- Testing Permuted_output_Dataset ---")

    # Get an original sample to compare against
    original_img, original_label = mnist_train[0]
    print(f"Original label for first sample: {original_label}")

    # 1. Test with a trivial permutation (identity)
    print("\n1. Testing with trivial permutation...")
    trivial_permutation = np.arange(data_config.num_classes)
    permuted_output_dataset_trivial = Permuted_output_Dataset(mnist_train, permutation=trivial_permutation)
    img_trivial, label_trivial = permuted_output_dataset_trivial[0]
    print(f"Label with trivial permutation: {label_trivial}")
    assert original_label == label_trivial

    # 2. Test with a non-trivial permutation (reversed)
    print("\n2. Testing with non-trivial permutation...")
    nontrivial_permutation = np.arange(data_config.num_classes)[::-1].copy() # reverse order
    permuted_output_dataset_nontrivial = Permuted_output_Dataset(mnist_train, permutation=nontrivial_permutation)
    
    # test 10 images:
    for i in range(10):
        original_img, original_label = mnist_train[i]
        img_nontrivial, label_nontrivial = permuted_output_dataset_nontrivial[i]
        print(f"Label with non-trivial permutation (original: {original_label}): {label_nontrivial}")
        assert label_nontrivial == nontrivial_permutation[original_label]
    print("Permuted_output_Dataset tests passed!")
    
    # 3. test with a custom permutation
    print("\n3. Testing with custom permutation...")
    custom_permutation = np.array([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])  #
    permuted_output_dataset_custom = Permuted_output_Dataset(mnist_train, permutation=custom_permutation)
    for i in range(10):
        original_img, original_label = mnist_train[i]
        img_custom, label_custom = permuted_output_dataset_custom[i]
        print(f"Label with custom permutation (original: {original_label}): {label_custom}")
        assert label_custom == custom_permutation[original_label]
    print("Permuted_output_Dataset custom permutation test passed!")
        
        