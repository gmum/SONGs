"""Wrappers around CIFAR datasets"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

__all__ = names = ("CIFAR10", "CIFAR100", "TinyImagenet200")


class InverseNormalize:
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[None, :, None, None]
        self.std = torch.Tensor(std)[None, :, None, None]

    def __call__(self, sample):
        return (sample * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class CIFAR:
    @staticmethod
    def transform_train():
        return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    @staticmethod
    def transform_val():
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    @staticmethod
    def transform_val_inverse():
        return InverseNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


class CIFAR10(datasets.CIFAR10, CIFAR):
    pass


class CIFAR100(datasets.CIFAR100, CIFAR):
    pass


class TinyImagenet200(Dataset):
    """Tiny imagenet dataloader"""

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    dataset = None

    def __init__(self, root="./data", *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            self.download(root=root)
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.targets = self.dataset.targets

    @staticmethod
    def transform_train(input_size=64):
        return transforms.Compose(
            [
                transforms.RandomCrop(input_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )

    @staticmethod
    def transform_val(input_size=-1):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )

    def download(self, root="./"):
        """Download and unzip Imagenet200 files in the `root` directory."""
        dir = os.path.join(root, "tiny-imagenet-200")
        dir_train = os.path.join(dir, "train")
        if os.path.exists(dir) and os.path.exists(dir_train):
            print("==> Already downloaded.")
            return

        path = Path(os.path.join(root, "tiny-imagenet-200.zip"))
        if not os.path.exists(path):
            os.makedirs(path.parent, exist_ok=True)

            print("==> Downloading TinyImagenet200...")
            with urllib.request.urlopen(self.url) as response, open(
                    str(path), "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

        print("==> Extracting TinyImagenet200...")
        with zipfile.ZipFile(str(path)) as zf:
            zf.extractall(root)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/train"), *args, **kwargs)
        self.targets = self.targets = [img[1] for img in self.imgs]


class _TinyImagenet200Val(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/val"), *args, **kwargs)

        self.path_to_class = {}
        with open(os.path.join(self.root, "val_annotations.txt")) as f:
            for line in f.readlines():
                parts = line.split()
                path = os.path.join(self.root, "images", parts[0])
                self.path_to_class[path] = parts[1]

        self.classes = list(sorted(set(self.path_to_class.values())))
        self.class_to_idx = {label: self.classes.index(label) for label in self.classes}
        self.targets = [self.class_to_idx[self.path_to_class[img[0]]] for img in self.imgs]

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        path, _ = self.samples[i]
        label = self.path_to_class[path]
        target = self.class_to_idx[label]
        return sample, target

    def __len__(self):
        return super().__len__()
