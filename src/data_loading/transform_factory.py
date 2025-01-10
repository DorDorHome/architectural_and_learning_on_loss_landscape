# The transform factory takes the dataset name and model name as inputs and returns the appropriate transformation pipeline.
# This keeps the logic for transformations isolated from both the dataset and model factories.

import torchvision.transforms as transforms

# for debugging, import the dataset_factory:

import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))



def transform_factory(dataset_name: str, model_name: str):
    """
    Creates a transform pipeline based on the dataset and model requirements.

    Args:
        dataset_name (str): Name of the dataset (e.g., CIFAR10, MNIST, ImageNet).
        model_name (str): Name of the model (e.g., ResNet18, VGG16).

    Returns:
        torchvision.transforms.Compose: A transform pipeline.
    """
    if dataset_name == "CIFAR10":
        if model_name == "ResNet18" or model_name == "resnet_custom":
            # ResNet18 expects normalized inputs
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif model_name == "VGG16" or model_name == "vgg_custom":
            # VGG16 may expect slightly different preprocessing
            return transforms.Compose([
                transforms.Resize(224),  # Resize to match VGG input size
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unsupported model {model_name} for dataset {dataset_name}")
    
    elif dataset_name == "ImageNet":
        if model_name == "ResNet18" or model_name == "resnet_custom":
            # ImageNet models expect normalized inputs
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        # ImageNet-specific transforms

        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]
        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                            ])
                            
    elif dataset_name == "MNIST":
        pass
        # return transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))  # MNIST-specific normalization
        # ])
    
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")
    
def test_transform_factory():
    
    
    transform = transform_factory("CIFAR10", "ResNet18")
    assert isinstance(transform, transforms.Compose)
    #print the transform pipeline:
    print(transform)
    
if __name__ == "__main__":
    # from src.data_loading.dataset_factory import dataset_factory
    
    test_transform_factory()
    
    
    