from torchvision import datasets, transforms
from torch.utils.data import Dataset

def get_dataset(name="mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])
    if name == "mnist":
        return datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")