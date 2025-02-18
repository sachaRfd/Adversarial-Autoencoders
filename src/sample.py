"""
Code to sample from a model.
"""

import argparse
import torch
from models import MNISTdecoder, MNISTencoder
import os
import matplotlib.pyplot as plt


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model"
    )
    args = parser.parse_args()
    return args


def load_model(model_class, optimizer_class, path, device="cpu"):
    """Loads the model, optimizer, and args."""
    checkpoint = torch.load(path, map_location=device)
    args = argparse.Namespace(**checkpoint["args"])  # Convert dict back to Namespace
    model = model_class(args=args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optimizer_class(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, args, epoch


def sample():
    args = load_args()

    # Load models
    encoder, _, enc_args, _ = load_model(
        MNISTencoder, torch.optim.Adam, os.path.join(args.model_path, "Encoder.pt")
    )
    decoder, _, dec_args, _ = load_model(
        MNISTdecoder, torch.optim.Adam, os.path.join(args.model_path, "Decoder.pt")
    )

    print("Models loaded with args:", enc_args, dec_args)
    encoder.eval()
    decoder.eval()

    # Example: Generate samples
    sample_input = torch.randn(1, enc_args.dim)  # Sample from latent space
    with torch.no_grad():
        generated = decoder(sample_input)
    print("Generated sample shape:", generated.shape)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    axes.imshow(generated.view(28, 28).numpy(), cmap="gray")
    axes.set_title("Generated")
    axes.axis("off")

    plt.show()


if __name__ == "__main__":
    sample()
