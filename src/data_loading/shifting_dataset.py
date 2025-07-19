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
import torch.nn.functional as F
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


class Slow_drift_contextual_bandit(Dataset[Tuple[Any, torch.Tensor]]):
    """
    A dataset wrapper that simulates a contextual bandit with slowly drifting true values.

    Args:
        original_dataset (Dataset): The base classification dataset.
        num_classes (int): The number of output classes.
        drift_mode (str): The type of drift to apply. Options are:
            - 'random_walk': Values drift via Brownian motion.
            - 'sinusoidal': Values oscillate based on a sine wave.
            - 'interpolation': Values gradually shift from a start to an end state.
        variance (float): Variance for the Gaussian noise added to the true value for feedback.
        **kwargs: Keyword arguments specific to the chosen drift mode.
            - For 'random_walk':
                - drift_std_dev (float): Std dev for the random walk noise. Default: 0.01.
            - For 'sinusoidal':
                - amplitude (float): The magnitude of the oscillation. Default: 0.5.
                - frequency (float): The speed of the oscillation (lower is slower). Default: 0.1.
            - For 'interpolation':
                - start_values (torch.Tensor): The initial set of true values.
                - end_values (torch.Tensor): The target set of true values.
                - duration (int): The number of updates (e.g., epochs) to complete the drift.
    """
    def __init__(self, original_dataset: Dataset,
                 num_classes: int,
                 drift_mode: str,
                 variance: float = 0.5,
                 **kwargs):
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.variance = variance
        self.drift_mode = drift_mode
        self.drift_params = kwargs
        
        # Track the current time step (e.g., epoch) for drift calculation
        self.time_step = 0

        # Initialize true values and validate parameters based on the mode
        self._initialize_drift()

    def _initialize_drift(self):
        """Sets up the initial state based on the selected drift mode."""
        if self.drift_mode == 'random_walk':
            self.drift_params.setdefault('drift_std_dev', 0.01)
            self.true_values = torch.randn(self.num_classes)
        
        elif self.drift_mode == 'sinusoidal':
            self.drift_params.setdefault('amplitude', 0.5)
            self.drift_params.setdefault('frequency', 0.1)
            self.base_values = torch.randn(self.num_classes)
            self.true_values = self.base_values.clone()
        
        elif self.drift_mode == 'interpolation':
            if not all(k in self.drift_params for k in ['start_values', 'end_values', 'duration']):
                raise ValueError("For 'interpolation' mode, 'start_values', 'end_values', and 'duration' must be provided.")
            if self.drift_params['duration'] <= 0:
                raise ValueError("'duration' must be a positive integer.")
            self.true_values = self.drift_params['start_values'].clone()
        
        else:
            raise ValueError(f"Unknown drift_mode: '{self.drift_mode}'")

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """Returns the image and a noisy reward based on the true label."""
        img, label = self.original_dataset[index]
        
        # Get the true value for the correct class
        underlying_value = self.true_values[label]
        
        # Draw from a Gaussian centered at the true value
        feedback_value = torch.normal(mean=underlying_value, std=self.variance**0.5)
        
        return img, feedback_value

    def __len__(self) -> int:
        return len(self.original_dataset)

    def update_drift(self):
        """
        Updates the true values based on the configured drift mode.
        This should be called periodically (e.g., once per epoch).
        """
        # Dispatch to the correct private update method
        update_method = getattr(self, f"_update_{self.drift_mode}")
        update_method()
        self.time_step += 1

    def _update_random_walk(self):
        """Drift values using a random walk."""
        std_dev = self.drift_params['drift_std_dev']
        noise = torch.normal(mean=0.0, std=std_dev, size=(self.num_classes,))
        self.true_values += noise

    def _update_sinusoidal(self):
        """Drift values using a sine wave."""
        amplitude = self.drift_params['amplitude']
        frequency = self.drift_params['frequency']
        # Use different phases for each value to desynchronize them
        phases = torch.linspace(0, 2 * np.pi, self.num_classes)
        drift = amplitude * torch.sin(frequency * self.time_step + phases)
        self.true_values = self.base_values + drift

    def _update_interpolation(self):
        """Drift values by interpolating between two points."""
        duration = self.drift_params['duration']
        if self.time_step >= duration:
            # Clamp to the end value after the duration has passed
            self.true_values = self.drift_params['end_values'].clone()
            return
            
        start = self.drift_params['start_values']
        end = self.drift_params['end_values']
        
        # Calculate the interpolation factor (alpha)
        # The number of steps is duration, so we interpolate over [0, duration-1].
        # To reach the end value at the last step, alpha should be 1.
        alpha = self.time_step / (duration - 1) if duration > 1 else 1.0
        
        # Linearly interpolate
        self.true_values = (1 - alpha) * start + alpha * end



class DriftingInputDataset(Dataset[Tuple[torch.Tensor, Any]]):
    def __init__(self, 
                 original_dataset: Dataset, 
                 drift_mode: str = 'affine',
                 **kwargs):
        """
        A dataset wrapper that applies a continuously drifting transformation to the input space.

        Args:
            original_dataset (Dataset): The base dataset (e.g., MNIST, CIFAR10).
            drift_mode (str): The type of drift. Currently supports 'affine'.
            **kwargs: Keyword arguments for the drift mode.
                - For 'affine':
                    - drift_type (str): 'random_walk' or 'sinusoidal'.
                    - rotation_std_dev / rotation_amplitude (float): Controls drift of angle.
                    - scale_std_dev / scale_amplitude (float): Controls drift of scale.
                    - shear_std_dev / shear_amplitude (float): Controls drift of shear.
                    # Add other parameters for frequency etc. as needed.
        """
        self.original_dataset = original_dataset
        self.drift_mode = drift_mode
        self.drift_params = kwargs
        self.time_step = 0

        if self.drift_mode != 'affine':
            raise NotImplementedError("Only 'affine' drift_mode is currently implemented.")
        
        self._initialize_affine_params()

    def _initialize_affine_params(self):
        """Initializes the parameters for the affine transformation."""
        self.drift_type = self.drift_params.get('drift_type', 'random_walk')
        
        # Store the current state of the transformation parameters
        self.transform_params = {
            'angle': 0.0,
            'scale': 1.0,
            'shear_x': 0.0,
            'shear_y': 0.0,
            'trans_x': 0.0,
            'trans_y': 0.0
        }
        
        # Store the parameters that control the drift itself
        self.drift_config = {
            'rotation': self.drift_params.get('rotation_std_dev', 2.0), # degrees
            'scale': self.drift_params.get('scale_std_dev', 0.01),
            'shear': self.drift_params.get('shear_std_dev', 0.02)
        }

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Retrieves an item and applies the current drifting transformation.
        """
        img, label = self.original_dataset[index]

        # Ensure image is a tensor with a batch dimension [N, C, H, W] for grid_sample
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        
        # Add batch and channel dimensions if missing
        if img.dim() == 2: # H, W -> 1, 1, H, W
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3: # C, H, W -> 1, C, H, W
            img = img.unsqueeze(0)

        # 1. Get the current transformation matrix
        theta = self._get_affine_matrix(img.device)

        # 2. Apply the affine transformation
        #    affine_grid generates a sampling grid
        #    grid_sample uses the grid to sample from the input image
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        drifted_img = F.grid_sample(img, grid, align_corners=False)

        # Remove the batch dimension before returning
        return drifted_img.squeeze(0), label

    def _get_affine_matrix(self, device: torch.device) -> torch.Tensor:
        """
        Constructs the 2x3 affine transformation matrix from current parameters.
        Note: F.affine_grid expects a matrix that maps output coordinates to source coordinates.
        Therefore, we must compute the inverse of our intuitive transformation.
        """
        p = self.transform_params
        angle_rad = np.deg2rad(p['angle'])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Forward transformation matrices (maps source to destination)
        rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        scale_matrix = torch.tensor([[p['scale'], 0], [0, p['scale']]], dtype=torch.float32)
        shear_matrix = torch.tensor([[1, p['shear_x']], [p['shear_y'], 1]], dtype=torch.float32)
        
        # Combined forward 2x2 transformation
        forward_transform_2x2 = shear_matrix @ scale_matrix @ rot_matrix
        
        # Forward translation vector
        forward_translation = torch.tensor([p['trans_x'], p['trans_y']], dtype=torch.float32)

        # Compute the inverse transformation
        # Inverse of the 2x2 matrix
        inverse_transform_2x2 = torch.inverse(forward_transform_2x2)
        # Inverse of the translation
        inverse_translation = -inverse_transform_2x2 @ forward_translation

        # Create the final 2x3 affine matrix for F.affine_grid
        transform_matrix = torch.zeros(2, 3, dtype=torch.float32)
        transform_matrix[:2, :2] = inverse_transform_2x2
        transform_matrix[:, 2] = inverse_translation
        
        return transform_matrix.unsqueeze(0).to(device) # Add batch dimension

    def update_drift(self):
        """Updates the transformation parameters based on the drift configuration."""
        if self.drift_type == 'random_walk':
            self._update_affine_random_walk()
        # Add 'sinusoidal' or other types here as needed
        
        self.time_step += 1

    def _update_affine_random_walk(self):
        """Applies a random walk to the affine parameters."""
        cfg = self.drift_config
        self.transform_params['angle'] += np.random.normal(0, cfg['rotation'])
        self.transform_params['scale'] += np.random.normal(0, cfg['scale'])
        # Clamp scale to prevent image from vanishing or inverting
        self.transform_params['scale'] = np.clip(self.transform_params['scale'], 0.5, 1.5)
        
        self.transform_params['shear_x'] += np.random.normal(0, cfg['shear'])
        self.transform_params['shear_y'] += np.random.normal(0, cfg['shear'])
        # You could also drift translation if desired
        
        print(f"Time {self.time_step}: Angle={self.transform_params['angle']:.2f}, Scale={self.transform_params['scale']:.2f}")

    def __len__(self) -> int:
        return len(self.original_dataset)



    
if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np
    from configs.configurations import DataConfig
    import os
    import torch.nn.functional as F
    from torchvision.utils import save_image

    # Create a directory for test outputs
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

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


    # --- Test for Slow_drift_contextual_bandit ---
    print("\n--- Testing Slow_drift_contextual_bandit ---")

    # 1. Test 'random_walk' mode with zero drift
    print("\n1. Testing 'random_walk' with zero drift...")
    bandit_rw = Slow_drift_contextual_bandit(
        mnist_train,
        num_classes=data_config.num_classes,
        drift_mode='random_walk',
        drift_std_dev=0.0  # Zero drift
    )
    initial_values_rw = bandit_rw.true_values.clone()
    bandit_rw.update_drift()
    print(f"Initial values: {initial_values_rw}")
    print(f"Values after 1 update: {bandit_rw.true_values}")
    assert torch.equal(initial_values_rw, bandit_rw.true_values), "Values should not change with zero drift."
    print("'random_walk' zero drift test passed!")

    # 2. Test 'sinusoidal' mode with a specific check
    print("\n2. Testing 'sinusoidal' drift logic...")
    amp = 0.5
    freq = 0.1
    bandit_sin = Slow_drift_contextual_bandit(
        mnist_train,
        num_classes=data_config.num_classes,
        drift_mode='sinusoidal',
        amplitude=amp,
        frequency=freq
    )
    base_values = bandit_sin.base_values.clone()

    # --- Check step 1 (uses time_step = 0) ---
    bandit_sin.update_drift()
    phases = torch.linspace(0, 2 * np.pi, data_config.num_classes)
    expected_drift_1 = amp * torch.sin(freq * 0 + phases)
    expected_values_1 = base_values + expected_drift_1
    
    print(f"Values after 1 update: {bandit_sin.true_values}")
    print(f"Expected values:       {expected_values_1}")
    assert torch.allclose(bandit_sin.true_values, expected_values_1), "Values should match expected sinusoidal drift for step 1."

    # --- Check step 2 (uses time_step = 1) ---
    bandit_sin.update_drift()
    expected_drift_2 = amp * torch.sin(freq * 1 + phases)
    expected_values_2 = base_values + expected_drift_2
    print(f"Values after 2 updates: {bandit_sin.true_values}")
    print(f"Expected values:        {expected_values_2}")
    assert torch.allclose(bandit_sin.true_values, expected_values_2), "Values should match expected sinusoidal drift for step 2."
    print("'sinusoidal' drift logic test passed!")

    # 3. Test 'interpolation' mode
    print("\n3. Testing 'interpolation' drift...")
    start_vals = torch.zeros(data_config.num_classes)
    end_vals = torch.ones(data_config.num_classes)
    duration = 10
    bandit_interp = Slow_drift_contextual_bandit(
        mnist_train,
        num_classes=data_config.num_classes,
        drift_mode='interpolation',
        start_values=start_vals,
        end_values=end_vals,
        duration=duration
    )
    
    # Check initial state
    assert torch.equal(bandit_interp.true_values, start_vals), "Should start at start_values."
    
    # Update to halfway point
    for _ in range(duration // 2):
        bandit_interp.update_drift()
    
    midpoint_values = bandit_interp.true_values.clone()
    print(f"Values at midpoint (step {bandit_interp.time_step}): {midpoint_values}")
    # After duration//2 steps, the time_step used for the last calculation was (duration//2 - 1).
    # The alpha should be (duration//2 - 1) / (duration - 1) if we start from time_step 0.
    # However, since update_drift increments time_step *after* calculation, the last calculation
    # used time_step = 4. So alpha = 4 / 9.
    expected_midpoint_val = 4/9
    assert torch.allclose(midpoint_values, torch.full((data_config.num_classes,), expected_midpoint_val)), "Should be at the correct interpolated value."

    # Update to the end
    # We already did 5 steps, so we need 5 more to reach step 9 for the final calculation
    for _ in range(duration - (duration // 2)):
        bandit_interp.update_drift()
        
    print(f"Values at the end (step {bandit_interp.time_step}): {bandit_interp.true_values}")
    assert torch.allclose(bandit_interp.true_values, end_vals), "Should finish at end_values."
    print("'interpolation' drift test passed!")


    # --- Test for DriftingInputDataset ---
    print("\n--- Testing DriftingInputDataset ---")

    # Get a sample image to work with
    original_img_tensor, _ = mnist_train[0]

    # 1. Trivial Test: No drift applied initially
    print("\n1. Testing initial state (identity transform)...")
    drift_dataset_trivial = DriftingInputDataset(mnist_train)
    transformed_img_trivial, _ = drift_dataset_trivial[0]
    
    # The initial transform should be identity, so images should be identical
    assert torch.allclose(original_img_tensor, transformed_img_trivial, atol=1e-5), "Initial transform should be identity."
    print("Initial state test passed!")

    def test_affine_drift_and_inversion(dataset, drift_params, test_name, image_idx=0):
        print(f"\n--- Running non-trivial test: '{test_name}' ---")
        
        # --- Setup ---
        drift_dataset = DriftingInputDataset(dataset, drift_type='random_walk', **drift_params)
        
        # Manually set the transformation parameters for a deterministic test
        drift_dataset.transform_params['angle'] = drift_params.get('angle', 0.0)
        drift_dataset.transform_params['scale'] = drift_params.get('scale', 1.0)
        drift_dataset.transform_params['shear_x'] = drift_params.get('shear_x', 0.0)
        drift_dataset.transform_params['shear_y'] = drift_params.get('shear_y', 0.0)
        
        original_img, _ = dataset[image_idx]
        transformed_img, _ = drift_dataset[image_idx]

        # --- Save images for visual inspection ---
        param_str = f"angle={drift_dataset.transform_params['angle']}_scale={drift_dataset.transform_params['scale']:.2f}"
        save_image(original_img, os.path.join(output_dir, f"{test_name}_original_image.png"))
        save_image(transformed_img, os.path.join(output_dir, f"{test_name}_transformed_{param_str}.png"))
        print(f"Saved original and transformed images for '{test_name}' in '{output_dir}/'")

        # --- Compute Inverse Transform and Verify ---
        # Get the forward transformation matrix used by the dataset
        forward_theta = drift_dataset._get_affine_matrix(device='cpu') # Shape: [1, 2, 3]

        # Separate the 2x2 matrix (A) and the translation vector (t)
        A = forward_theta[:, :2, :2]
        t = forward_theta[:, :2, 2].unsqueeze(2)

        # Compute the inverse of the 2x2 matrix A
        A_inv = torch.inverse(A)

        # Compute the new translation vector for the inverse transform: -A_inv * t
        t_inv = -A_inv @ t

        # Assemble the inverse transformation matrix
        inverse_theta = torch.cat([A_inv, t_inv], dim=2)

        # Apply the inverse transformation to the transformed image
        # Add batch dim for grid_sample
        grid = F.affine_grid(inverse_theta, transformed_img.unsqueeze(0).size(), align_corners=False)
        reconstructed_img = F.grid_sample(transformed_img.unsqueeze(0), grid, align_corners=False).squeeze(0)

        # --- Assert ---
        # The reconstructed image should be very close to the original
        mse = F.mse_loss(reconstructed_img, original_img)
        print(f"MSE between original and reconstructed image: {mse.item()}")
        assert mse < 1e-2, f"Reconstruction failed for '{test_name}'. MSE: {mse.item()}"
        print(f"Inverse transform test passed for '{test_name}'!")

    # 2. Non-trivial Test: Small Displacement
    test_affine_drift_and_inversion(
        mnist_train,
        {'angle': 15.0, 'scale': 1.1, 'shear_x': 0.1, 'shear_y': -0.1},
        "small_displacement"
    )

    # 3. Non-trivial Test: Large Displacement
    test_affine_drift_and_inversion(
        mnist_train,
        {'angle': -45.0, 'scale': 0.7, 'shear_x': -0.3, 'shear_y': 0.4},
        "large_displacement"
    )

