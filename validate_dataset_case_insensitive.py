"""Quick validation for Step 5: case-insensitive dataset handling.

We attempt to load datasets using lowercase names and ensure that both the
transform_factory and dataset_factory succeed and return objects.

This script should run fast: it only instantiates the datasets (will download if
not present). If download is heavy, user may interrupt after confirming the
normalization warning appears.
"""
from configs.configurations import DataConfig
from src.data_loading.transform_factory import transform_factory
from src.data_loading.dataset_factory import dataset_factory


def try_dataset(name: str):
    print(f"\n=== Attempting dataset '{name}' ===")
    cfg = DataConfig(dataset=name, data_path="/hdda/datasets", use_torchvision=True)
    transform = transform_factory(name, model_name="ResNet18")
    trainset, testset = dataset_factory(cfg, transform=transform, with_testset=True)
    print(f"Trainset type: {type(trainset)}; length={len(trainset)}")
    if testset is not None:
        print(f"Testset type: {type(testset)}; length={len(testset)}")
    print("SUCCESS: case-insensitive load works.")


def main():
    for ds in ["cifar10", "mnist"]:
        try_dataset(ds)

if __name__ == "__main__":
    main()
