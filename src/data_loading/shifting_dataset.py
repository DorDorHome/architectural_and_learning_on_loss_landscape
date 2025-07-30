# This file contains classes useful for shifting the dataset.

from typing import Any, Dict, Tuple, Union, Optional, Callable, List, Sized
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms

from configs.configurations import ExperimentConfig


class Permuted_input_Dataset(Dataset[Tuple[Any, Any]]):
    def __init__(self, original_dataset: Dataset[Any], permutation: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None, flatten: bool = False, transform: Optional[Callable[..., Any]] = None):
        super().__init__()
        self.original_dataset = original_dataset
        self.flatten = flatten
        self.transform = transform
        self.permutation: torch.Tensor

        # Determine the shape of the input data from the first sample
        sample_img, _ = self.original_dataset[0]
        
        # Handle both torch tensors and PIL Images
        if hasattr(sample_img, 'shape'):
            original_shape = sample_img.shape
        elif hasattr(sample_img, 'size'):
            original_shape = (1, sample_img.size[1], sample_img.size[0]) # Assuming CHW for PIL
        else:
            raise TypeError("Unsupported image type. Expected a PIL Image or a torch Tensor.")

        if self.flatten:
            # For flattened input, the size is the total number of elements
            expected_size = int(np.prod(original_shape))
        else:
            # For non-flattened, permutation is on the spatial dimensions (H*W)
            if len(original_shape) < 3:
                raise ValueError("Expected image with at least 3 dimensions (C, H, W) for spatial permutation.")
            expected_size = original_shape[1] * original_shape[2]

        # Validate and set up the permutation
        if permutation is None:
            self.permutation = torch.randperm(expected_size)
        else:
            # Ensure permutation is a tensor
            if not isinstance(permutation, torch.Tensor):
                permutation = torch.tensor(permutation, dtype=torch.long)
            
            if len(permutation) != expected_size:
                raise ValueError(f"Permutation size {len(permutation)} must match the expected size {expected_size}")
            self.permutation = permutation

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.original_dataset[index]

        if self.transform:
            img = self.transform(img)

        # Ensure image is a tensor before permuting
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        if self.permutation is not None:
            if self.flatten:
                # Flatten the image and then permute
                flat_img = img.view(-1)
                permuted_img = flat_img[self.permutation]
                # Reshape back to the original image shape
                img = permuted_img.view(img.shape)
            else:
                # Permute spatial dimensions (H*W) for each channel
                c, h, w = img.shape
                img_reshaped = img.view(c, -1)
                if len(self.permutation) != h * w:
                    raise ValueError(f"Permutation size {len(self.permutation)} must match the spatial size {h*w} for non-flattened input.")
                img_reshaped = img_reshaped[:, self.permutation]
                img = img_reshaped.view(c, h, w)

        return img, label

    def __len__(self) -> int:
        if isinstance(self.original_dataset, Sized):
            return len(self.original_dataset)
        raise TypeError("Original dataset does not have a __len__ method.")


class Permuted_output_Dataset(Dataset[Tuple[Any, int]]):
    def __init__(self, original_dataset: Dataset[Any], permutation: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None):
        super().__init__()
        self.original_dataset = original_dataset
        self.permutation: torch.Tensor
        
        # Infer num_classes from the dataset
        if hasattr(original_dataset, 'classes') and original_dataset.classes is not None:
             self.num_classes = len(original_dataset.classes)
        else:
            # Fallback for datasets without a .classes attribute
            print("Warning: .classes attribute not found on dataset. Inferring from unique labels.")
            labels = [label for _, label in original_dataset]
            self.num_classes = len(set(labels))
            print(f"Inferred {self.num_classes} classes.")


        if permutation is None:
            # Generate a random permutation if none is provided
            self.permutation = torch.randperm(self.num_classes)
        else:
            # Use the provided permutation
            if not isinstance(permutation, torch.Tensor):
                self.permutation = torch.tensor(permutation, dtype=torch.long)
            else:
                self.permutation = permutation

        # Validate permutation
        if len(self.permutation) != self.num_classes:
            raise ValueError(f"Permutation length {len(self.permutation)} must match number of classes {self.num_classes}")

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img, label = self.original_dataset[index]
        permuted_label = self.permutation[label].item()
        return img, int(permuted_label)

    def __len__(self) -> int:
        if isinstance(self.original_dataset, Sized):
            return len(self.original_dataset)
        raise TypeError("Original dataset does not have a __len__ method.")


class ContinuousDeformationDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    A stateful dataset wrapper that applies a continuous, time-varying affine transformation
    to the input images. The transformation evolves with each call to `update_task`.
    """
    def __init__(self, 
                 original_dataset: Dataset[Any], 
                 mode: str, 
                 num_classes: int,
                 transform_params: Dict[str, Any],
                 seed: Optional[int]):
        self.original_dataset = original_dataset
        self.mode = mode
        self.num_classes = num_classes
        self.transform_params = transform_params
        self.seed = seed
        self.time_step: int = 0
        self.theta: torch.Tensor = self._generate_random_theta()
        self.drift_velocity: Optional[torch.Tensor] = None

        if self.mode == 'linear':
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed + 1) # Use a different seed to avoid correlation
            # Define a random but fixed direction for the drift
            self.drift_velocity = torch.empty(2, 3).uniform_(-1, 1, generator=g)
            # Normalize to create a unit vector for direction
            norm_val = torch.norm(self.drift_velocity)
            if norm_val > 0:
                self.drift_velocity /= norm_val

    def _generate_random_theta(self) -> torch.Tensor:
        """
        Generates a random affine transformation matrix (theta) on the CPU.
        Initializes with an identity matrix and adds small random noise.
        """
        g = torch.Generator()
        if self.seed is not None:
            g.manual_seed(self.seed)
        
        # Identity matrix for a 2x3 affine transformation
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        
        # Add small random noise for initial deformation
        noise = torch.empty(2, 3).uniform_(-0.1, 0.1, generator=g)
        theta += noise
        return theta

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.original_dataset[index]
        
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        # Add batch dimension, apply affine grid, then remove batch dimension
        image_unsqueezed = image.unsqueeze(0)
        grid = F.affine_grid(self.theta.unsqueeze(0), list(image_unsqueezed.shape), align_corners=False)
        image = F.grid_sample(image_unsqueezed, grid, align_corners=False).squeeze(0)
        
        return image, label

    def __len__(self) -> int:
        if isinstance(self.original_dataset, Sized):
            return len(self.original_dataset)
        raise TypeError("Original dataset does not have a __len__ method.")

    def update_task(self):
        """
        Updates the transformation matrix (theta) based on the deformation mode.
        This should be called once per task/epoch to evolve the transformation.
        """
        update_method_name = f"_update_{self.mode}"
        update_method = getattr(self, update_method_name, self._update_identity)
        update_method()
        self.time_step += 1

    def _update_identity(self):
        """No update, theta remains constant."""
        pass

    def _update_linear(self):
        """Update theta with a constant velocity drift."""
        if self.drift_velocity is not None:
            step_size = self.transform_params.get('max_drift', 0.01)
            self.theta += self.drift_velocity * step_size

    def _update_random_walk(self):
        """Update theta with a small random walk."""
        std_dev = self.transform_params.get('drift_std_dev', 0.01)
        noise = torch.normal(mean=0.0, std=std_dev, size=(2, 3))
        self.theta += noise

    def _update_sinusoidal(self):
        """Update theta with a sinusoidal drift."""
        amplitude = self.transform_params.get('amplitude', 0.01)
        frequency = self.transform_params.get('frequency', 0.1)
        
        # Create a base identity matrix
        base_theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        
        # Apply a sinusoidal wave to it
        drift = amplitude * torch.sin(torch.tensor(frequency * self.time_step, dtype=torch.float))
        self.theta = base_theta + drift


class DriftingValuesDataset(Dataset[Tuple[Any, torch.Tensor]]):
    """
    A dataset wrapper that simulates a regression task where the target values for each
    class drift over time. It ensures that the relative order of class values is
    maintained through a repulsion mechanism.

    1.  Initially, the value for class `i` is `i.0`.
    2.  Values undergo a slow random walk.
    3.  If values for adjacent classes collide (e.g., value[i] >= value[i+1]),
        a repulsion force pushes them apart to maintain order.
    """
    def __init__(self, original_dataset: Dataset[Any],
                 num_classes: int,
                 drift_std_dev: float = 0.01,
                 repulsion_strength: float = 0.5,
                 min_gap: float = 0.1,
                 lower_bound: float = -20.0,
                 upper_bound: float = 20.0):
        """
        Args:
            original_dataset (Dataset): The base classification dataset.
            num_classes (int): The number of classes.
            drift_std_dev (float): Std dev for the random walk noise.
            repulsion_strength (float): How strongly colliding values push each other apart.
            min_gap (float): The minimum gap to enforce between adjacent values after a collision.
            lower_bound (float): Minimum allowed value (prevents drift to -infinity).
            upper_bound (float): Maximum allowed value (prevents drift to +infinity).
        """
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.drift_std_dev = drift_std_dev
        self.repulsion_strength = repulsion_strength
        self.min_gap = min_gap
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Track the current time step (e.g., epoch) for drift calculation
        self.time_step: int = 0
        
        # 1. Initialize values: value for class i is i.0
        self.values = torch.arange(self.num_classes, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor, int]:
        """Returns the image, its current drifting target value, and the original label."""
        img, label = self.original_dataset[index]
        
        # Get the current drifting value for the corresponding class label
        target_value = self.values[label]
        
        return img, (target_value, label)

    def __len__(self) -> int:
        if isinstance(self.original_dataset, Sized):
            return len(self.original_dataset)
        raise TypeError("Original dataset does not have a __len__ method.")

    def update_drift(self):
        """
        Updates the target values with a random walk and applies a repulsion
        force to prevent and resolve collisions. This should be called
        periodically (e.g., once per epoch).
        """
        # 2. Apply a random walk to all values
        noise = torch.normal(mean=0.0, std=self.drift_std_dev, size=(self.num_classes,))
        self.values += noise

        # 3. Handle collisions with repulsion first
        # We loop multiple times to allow repulsion forces to propagate and settle.
        max_iterations = self.num_classes * 4  # More iterations for complex cases
        for iteration in range(max_iterations):
            collided = False
            # Check for collisions between all adjacent class values
            for i in range(self.num_classes - 1):
                # A collision occurs if a lower-indexed value crosses an upper-indexed one
                if self.values[i] >= self.values[i+1] - self.min_gap:
                    collided = True
                    # Calculate how much they have overlapped
                    overlap = self.values[i] - (self.values[i+1] - self.min_gap)
                    
                    # Apply a symmetric repulsion force
                    half_correction = (overlap + self.min_gap) * 0.5 * self.repulsion_strength
                    self.values[i] -= half_correction
                    self.values[i+1] += half_correction
            
            # If no collisions were detected in a full pass, the system is stable
            if not collided:
                break
        
        # Final safety check: if ordering is still violated, sort and space them
        if not torch.all(self.values[:-1] <= self.values[1:] - self.min_gap):
            # Sort the values and enforce minimum spacing
            sorted_values, _ = torch.sort(self.values)
            for i in range(1, len(sorted_values)):
                if sorted_values[i] < sorted_values[i-1] + self.min_gap:
                    sorted_values[i] = sorted_values[i-1] + self.min_gap
            self.values = sorted_values

        # 4. Enforce boundary constraints AFTER collision resolution
        # This ensures we maintain ordering while respecting boundaries
        
        # If the range exceeds available space, compress uniformly
        total_range = self.values.max() - self.values.min()
        available_space = self.upper_bound - self.lower_bound - (self.num_classes - 1) * self.min_gap
        
        if total_range > available_space:
            # Compress the values to fit within bounds
            compression_factor = available_space / total_range
            center = (self.values.max() + self.values.min()) * 0.5
            self.values = center + (self.values - center) * compression_factor
        
        # Check if the minimum value hits the lower bound
        if self.values.min() < self.lower_bound:
            # Shift all values up to respect the lower bound
            shift_up = self.lower_bound - self.values.min()
            self.values += shift_up
        
        # Check if the maximum value hits the upper bound
        if self.values.max() > self.upper_bound:
            # Shift all values down to respect the upper bound
            shift_down = self.values.max() - self.upper_bound
            self.values -= shift_down
        
        self.time_step += 1


# --- Dataset Factories ---

def create_stateless_dataset_wrapper(cfg: ExperimentConfig, base_dataset: Dataset[Any], task_idx: int) -> Dataset[Any]:
    """Creates a stateless dataset wrapper based on the configuration for a specific task."""
    shift_mode = getattr(cfg, 'task_shift_mode', 'no_shift')

    if shift_mode == 'permuted_input':
        g = torch.Generator()
        seed = getattr(cfg, 'seed', None)
        if seed is not None:
            g.manual_seed(seed + task_idx)
        
        sample_img, _ = base_dataset[0]
        if hasattr(sample_img, 'shape'):
            num_features = int(np.prod(sample_img.shape))
        else: # Handle PIL images
            num_features = int(np.prod((1, sample_img.size[1], sample_img.size[0])))
        permutation = torch.randperm(num_features, generator=g).tolist()
        
        return Permuted_input_Dataset(base_dataset, permutation=permutation, flatten=True)

    elif shift_mode == 'permuted_output':
        g = torch.Generator()
        seed = getattr(cfg, 'seed', None)
        if seed is not None:
            g.manual_seed(seed + task_idx)
        
        data_cfg = getattr(cfg, 'data', None)
        num_classes = getattr(data_cfg, 'num_classes', None) if data_cfg else None
        if num_classes is None:
            raise ValueError("cfg.data.num_classes must be defined for permuted_output")
        permutation = torch.randperm(num_classes, generator=g).tolist()
        return Permuted_output_Dataset(base_dataset, permutation=permutation)

    else:
        return base_dataset

def create_stateful_dataset_wrapper(cfg: ExperimentConfig, train_set: Dataset[Any]) -> Dataset[Any]:
    """Creates a stateful dataset wrapper based on the configuration."""
    task_shift_mode = getattr(cfg, 'task_shift_mode', None)
    task_shift_param = getattr(cfg, 'task_shift_param', None)

    if not task_shift_mode or not task_shift_param:
        return train_set

    if task_shift_mode == 'drifting_values': # Renamed from 'slow_drift_bandit'
        params = getattr(task_shift_param, 'drifting_values', None)
        if not params:
            raise ValueError("Missing 'drifting_values' params in config")
        
        data_cfg = getattr(cfg, 'data', None)
        num_classes = getattr(data_cfg, 'num_classes', None) if data_cfg else None
        if num_classes is None:
            raise ValueError("cfg.data.num_classes must be defined for drifting_values")

        return DriftingValuesDataset(
            original_dataset=train_set,
            num_classes=num_classes,
            drift_std_dev=params.drift_std_dev,
            repulsion_strength=params.repulsion_strength,
            min_gap=params.min_gap,
            lower_bound=params.value_bounds.lower_bound,
            upper_bound=params.value_bounds.upper_bound
        )
    elif task_shift_mode == 'continuous_input_deformation':
        params = getattr(task_shift_param, 'continuous_input_deformation', None)
        if not params:
            raise ValueError("Missing 'continuous_input_deformation' params in config")

        drift_mode = getattr(params, 'drift_mode', None)
        if not drift_mode:
            raise ValueError("Missing 'drift_mode' in continuous_input_deformation params")

        transform_params_dict = OmegaConf.to_container(getattr(params, drift_mode, {}), resolve=True)
        
        data_cfg = getattr(cfg, 'data', None)
        num_classes = getattr(data_cfg, 'num_classes', None) if data_cfg else None
        if num_classes is None:
            raise ValueError("cfg.data.num_classes must be defined for continuous_input_deformation")
        
        seed = getattr(cfg, 'seed', None)

        return ContinuousDeformationDataset(
            original_dataset=train_set,
            mode=drift_mode,
            num_classes=num_classes,
            transform_params=transform_params_dict,
            seed=seed
        )
    return train_set




if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from PIL import Image
    import numpy as np
    from configs.configurations import DataConfig
    import os
    import torch.nn.functional as F
    from torchvision.utils import save_image
    from .dataset_factory import dataset_factory
    
    

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


    # --- Test for ContinuousDeformationDataset ---
    print("\n--- Testing ContinuousDeformationDataset ---")
    
    # Setup a mock base dataset (e.g., MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    try:
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    except Exception as e:
        print(f"Could not download MNIST, creating a dummy dataset. Error: {e}")
        # Create a dummy dataset if MNIST download fails
        dummy_data = torch.randn(100, 1, 28, 28)
        dummy_labels = torch.randint(0, 10, (100,))
        mnist_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)


    num_tasks = 10
    # Test linear drift
    print("\nTesting Linear Drift...")
    drift_dataset_linear = ContinuousDeformationDataset(
        original_dataset=mnist_dataset,
        num_tasks=num_tasks,
        drift_mode='linear',
        max_drift=0.5,
        seed=42
    )
    
    # Check initial transformation (should be near identity)
    initial_img, _ = drift_dataset_linear[0]
    base_img, _ = mnist_dataset[0]
    
    # The first task is already updated, so it's not pure identity
    # Let's check the inverse transformation
    inv_theta = drift_dataset_linear.get_inverse_transformation()
    
    # Reconstruct the image
    inv_affine_grid = F.affine_grid(inv_theta.unsqueeze(0), initial_img.unsqueeze(0).size(), align_corners=False).to(drift_dataset_linear.device)
    reconstructed_img = F.grid_sample(initial_img.unsqueeze(0), inv_affine_grid, align_corners=False)
    
    # It should be close to the original
    reconstruction_error = torch.mean((reconstructed_img.squeeze(0) - base_img.to(drift_dataset_linear.device))**2)
    print(f"Reconstruction error after initial linear step: {reconstruction_error.item()}")
    assert reconstruction_error < 1e-2, "Reconstruction failed for linear drift."

    # Update task and check again
    drift_dataset_linear.update_task() # Move to task 2
    img_task2, _ = drift_dataset_linear[0]
    
    # Test sinusoidal drift
    print("\nTesting Sinusoidal Drift...")
    drift_dataset_sin = ContinuousDeformationDataset(
        original_dataset=mnist_dataset,
        num_tasks=num_tasks,
        drift_mode='sinusoidal',
        max_drift=0.8,
        seed=123
    )
    img_sin_task1, _ = drift_dataset_sin[0]
    drift_dataset_sin.update_task()
    img_sin_task2, _ = drift_dataset_sin[0]
    
    # The images should be different
    assert not torch.equal(img_sin_task1, img_sin_task2), "Image should change between tasks in sinusoidal drift."
    print("Sinusoidal drift test passed.")

    # Test random walk drift
    print("\nTesting Random Walk Drift...")
    drift_dataset_rw = ContinuousDeformationDataset(
        original_dataset=mnist_dataset,
        num_tasks=num_tasks,
        drift_mode='random_walk',
        max_drift=0.1, # smaller drift for rw
        seed=99
    )
    img_rw_task1, _ = drift_dataset_rw[0]
    drift_dataset_rw.update_task()
    img_rw_task2, _ = drift_dataset_rw[0]
    assert not torch.equal(img_rw_task1, img_rw_task2), "Image should change between tasks in random walk drift."
    print("Random walk drift test passed.")

    print("\nAll ContinuousDeformationDataset tests passed!")


    # --- Test DriftingValuesDataset ---
    print("\n--- Testing DriftingValuesDataset ---")

    # 1. Test initialization
    print("\n1. Testing initialization...")
    dataset = DriftingValuesDataset(
        mnist_train,
        num_classes=data_config.num_classes,
        drift_std_dev=0.1,
        repulsion_strength=0.5,
        min_gap=0.1
    )
    expected_initial_values = torch.arange(data_config.num_classes, dtype=torch.float32)
    print(f"Initial values: {dataset.values}")
    print(f"Expected initial values: {expected_initial_values}")
    assert torch.equal(dataset.values, expected_initial_values), "Initial values are incorrect."
    print("Initialization test passed!")

    # 2. Test drift and order preservation
    print("\n2. Testing drift and order preservation...")
    initial_values = dataset.values.clone()
    # Update multiple times to see drift
    for i in range(10):
        dataset.update_drift()
        print(f"Values after update {i+1}: {dataset.values}")
        # Check that values have changed
        assert not torch.equal(initial_values, dataset.values), "Values should drift after update."
        # Check that order is preserved
        for j in range(data_config.num_classes - 1):
            assert dataset.values[j] < dataset.values[j+1], f"Order violated at index {j} after update {i+1}"
    print("Drift and order preservation test passed after 10 updates.")

    # 3. Test collision-repulsion mechanism
    print("\n3. Testing collision-repulsion mechanism...")
    # Engineer a collision
    dataset_for_collision_test = DriftingValuesDataset(
        mnist_train,
        num_classes=data_config.num_classes,
        drift_std_dev=0.0, # No random drift
        repulsion_strength=0.5,
        min_gap=0.1
    )
    # Manually create a collision
    dataset_for_collision_test.values[1] = 0.5
    dataset_for_collision_test.values[2] = 0.4 # Collision: value[1] > value[2]
    print(f"Values before repulsion: {dataset_for_collision_test.values}")

    # update_drift should fix this even without noise
    dataset_for_collision_test.update_drift()
    print(f"Values after repulsion: {dataset_for_collision_test.values}")
    assert dataset_for_collision_test.values[1] < dataset_for_collision_test.values[2], "Repulsion mechanism failed to correct collision."
    print("Collision-repulsion test passed!")

