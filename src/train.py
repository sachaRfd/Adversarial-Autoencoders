import argparse
from datasets import load_mnist
from models import MNISTencoder, MNISTdecoder
from utils import print_parameters, save_model
import torch
import torch.nn as nn
from tqdm import tqdm


def load_args():
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
    args = parser.parse_args()
    return args


def train():
    args = load_args()

    # Load the datasets:
    if args.dataset == "mnist":
        train_loader, test_loader = load_mnist(args)
    else:
        print("Dataset not specified correctly")
        print("choose --dataset <mnist>")
        return

    # Load models:
    encoder = MNISTencoder(args=args)
    decoder = MNISTdecoder(args=args)

    print_parameters(encoder)
    print_parameters(decoder)

    # Load optimizers:
    enc_optim = torch.optim.Adam(params=encoder.parameters(), lr=1e-4)
    dec_optim = torch.optim.Adam(params=decoder.parameters(), lr=1e-4)

    # Reconstruction loss
    ae_criterion = nn.MSELoss()

    for e in range(args.epochs):
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {e + 1}/{args.epochs}", leave=True
        )
        encoder.train()
        decoder.train()
        for i, (data, _) in enumerate(progress_bar):
            encoder.zero_grad()
            decoder.zero_grad()
            # Input:
            enc_out = encoder(data)
            dec_out = decoder(enc_out)

            loss = ae_criterion(dec_out.view(-1, 28, 28), data.view(-1, 28, 28))
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            progress_bar.set_postfix(loss=loss.item())

        # Save every 2 epochs:
        # if args.epochs % e == 0:
        save_model(
            encoder,
            enc_optim,
            e,
            args=args,
            path=f"models/{args.dataset}/epoch_{e}/Encoder",
        )
        save_model(
            decoder,
            dec_optim,
            e,
            args=args,
            path=f"models/{args.dataset}/epoch_{e}/Decoder",
        )


if __name__ == "__main__":
    train()
