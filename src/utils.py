"""
Random utils
"""

import torch
import os


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {model.__class__.__name__}: {num_params:,}")


def save_model(model, optimizer, epoch, args, path):
    """Saves the model, optimizer state, and args."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),  # Save args as a dictionary
    }
    torch.save(checkpoint, f"{path}.pt")
