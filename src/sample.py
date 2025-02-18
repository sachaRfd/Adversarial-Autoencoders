"""
Code to sample from a model.
"""

import argparse
import torch
from models import MNISTdecoder, MNISTencoder
from datasets import load_mnist
from args import load_sampling_args
import os
import matplotlib.pyplot as plt


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
    """
    Sample from the latent Space.
    """
    print("Sampling using Random Noise")
    args = load_sampling_args()

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
    axes.set_title("Generated with Random Noise")
    axes.axis("off")
    plt.show()


def reconstruct():
    """
    Check the reconstruction of some samples.
    """
    args = load_sampling_args()

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

    # Load Datasets:
    _, test_loader = load_mnist(enc_args)

    samples, _ = next(iter(test_loader))
    num_samples = 3  # Number of samples to reconstruct
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

    for i in range(num_samples):
        sample = samples[i]  # Select a specific sample
        enc_out = encoder(sample.unsqueeze(0))  # Add batch dimension (1, 1, 28, 28)
        reconstructed_sample = decoder(enc_out)

        # Plot original and reconstructed image side by side
        ax_original = axes[i, 0]
        ax_reconstructed = axes[i, 1]

        # Plot original image
        ax_original.imshow(sample.view(28, 28).numpy(), cmap="gray")
        ax_original.set_title(f"Original {i + 1}")
        ax_original.axis("off")

        # Plot reconstructed image
        ax_reconstructed.imshow(
            reconstructed_sample.view(28, 28).detach().numpy(), cmap="gray"
        )
        ax_reconstructed.set_title(f"Reconstructed {i + 1}")
        ax_reconstructed.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    reconstruct()
    sample()
