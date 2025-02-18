"""
Loading datasets
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(args):
    torch.cuda.manual_seed(1)
    path = "data/mnist"
    train_loader = DataLoader(
        datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batchsize,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batchsize,
        shuffle=False,
    )
    return train_loader, test_loader
