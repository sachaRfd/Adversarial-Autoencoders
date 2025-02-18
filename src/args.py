"""
Command line Arguments
"""

import argparse


def load_train_args():
    parser = argparse.ArgumentParser(description="aa-args")
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("-d", "--dim", type=int, default=64, help="latent space size")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of Training Epochs"
    )

    parser.add_argument(
        "--adversarial",
        type=bool,
        default=False,
        help="Whether or not to train with Adversarial loss.",
    )

    args = parser.parse_args()
    return args


def load_sampling_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model"
    )
    args = parser.parse_args()
    return args
