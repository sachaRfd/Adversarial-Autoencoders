"""

Training an Adversarial Autoencoder.

"""

import argparse
from datasets import load_mnist


def load_args():
    parser = argparse.ArgumentParser(description="aa-args")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--batchsize", default=128)
    args = parser.parse_args()
    return args


def train():
    args = load_args()

    # Load the datasets:
    if args.dataset == "mnist":
        train_loader, test_loader = load_mnist(args)
    else:
        raise ValueError("Please choose between: mnist/XX")

    sample = next(iter(train_loader))
    print(sample[0].shape, sample[1].shape)


if __name__ == "__main__":
    train()
