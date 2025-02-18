from datasets import load_mnist
from models import MNISTencoder, MNISTdecoder, MNISTdiscriminator
from utils import print_parameters, save_model
from args import load_train_args
import torch
import torch.nn as nn
from tqdm import tqdm


def train_reconstruction(args):
    print("Training an Autoencoder with reconstruction Loss")

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

    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(enc_optim, gamma=0.99)
    scheduler_dec = torch.optim.lr_scheduler.ExponentialLR(dec_optim, gamma=0.99)

    # Reconstruction loss
    ae_criterion = nn.MSELoss()

    for e in range(args.epochs):
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {e + 1}/{args.epochs}", leave=True
        )
        encoder.train()
        decoder.train()

        train_loss = 0.0
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
            train_loss += loss.item()

            scheduler_enc.step()
            scheduler_dec.step()

        train_loss /= len(train_loader)
        print(f"Epoch: {e}\t Train Loss: {train_loss}")

        # Save Intermediary Models:
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


def train_with_discriminator(args):
    """
    Train the Auto-Encoder with discriminator network.
    """

    print("Training an Autoencoder with Recon/Discr Loss")

    # Load the datasets:
    if args.dataset == "mnist":
        train_loader, test_loader = load_mnist(args)
    else:
        print("Dataset not specified correctly")
        print("choose --dataset <mnist>")
        exit()

    # Load models:
    encoder = MNISTencoder(args=args)
    decoder = MNISTdecoder(args=args)
    discriminator = MNISTdiscriminator(args=args)

    print_parameters(encoder)
    print_parameters(decoder)
    print_parameters(discriminator)

    # Load optimizers:
    enc_optim = torch.optim.Adam(params=encoder.parameters(), lr=1e-4)
    dec_optim = torch.optim.Adam(params=decoder.parameters(), lr=1e-4)
    disc_optim = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    # scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(enc_optim, gamma=0.99)
    # scheduler_dec = torch.optim.lr_scheduler.ExponentialLR(dec_optim, gamma=0.99)
    # scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(disc_optim, gamma=0.99)

    # Criterion:
    reconstruction_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    for e in range(args.epochs):
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {e + 1}/{args.epochs}", leave=True
        )

        train_recon_loss = 0.0
        train_disc_loss = 0.0
        train_confusion_loss = 0.0

        for i, (data, labels) in enumerate(progress_bar):
            # =========================
            # (1) Train Autoencoder
            # =========================
            encoder.train()
            decoder.train()
            discriminator.eval()
            for p in discriminator.parameters():
                p.requires_grad = False

            encoder.zero_grad()
            decoder.zero_grad()

            # Input:
            enc_out = encoder(data)
            dec_out = decoder(enc_out)

            loss = reconstruction_criterion(
                dec_out.view(-1, 28, 28), data.view(-1, 28, 28)
            )
            loss.backward()
            enc_optim.step()
            dec_optim.step()

            train_recon_loss += loss.item()

            # =========================
            # (2) Train Discriminator
            # =========================
            encoder.eval()
            decoder.eval()
            discriminator.train()

            for p in discriminator.parameters():
                p.requires_grad = True
            discriminator.zero_grad()

            # Encode real samples and Generate Fake samples:
            encoded_samples = encoder(data).detach()
            fake_samples = torch.randn_like(encoded_samples)  # Sample from prior

            # Discriminator output
            disc_real_output = discriminator(encoded_samples)
            disc_fake_output = discriminator(fake_samples)

            # Create labels
            real_labels = torch.ones_like(disc_real_output)
            fake_labels = torch.zeros_like(disc_fake_output)

            # Compute adversarial loss
            disc_loss = adversarial_criterion(
                disc_real_output, real_labels
            ) + adversarial_criterion(disc_fake_output, fake_labels)

            disc_loss.backward()
            disc_optim.step()

            train_disc_loss += disc_loss.item()

            # =========================
            # (3) Train Encoder
            # =========================
            encoder.train()
            for p in discriminator.parameters():
                p.requires_grad = False
            encoder.zero_grad()

            # Encode real samples:
            encoded_samples = encoder(data)

            # Discriminator output on real samples
            disc_real_output = discriminator(encoded_samples)

            # Encoder loss: try to fool the discriminator
            enc_loss = adversarial_criterion(disc_real_output, fake_labels)
            enc_loss.backward()
            enc_optim.step()

            train_confusion_loss += enc_loss.item()

            progress_bar.set_postfix(
                recon_loss=f"{train_recon_loss / (i + 1):.4f}",
                disc_loss=f"{train_disc_loss / (i + 1):.4f}",
                conf_loss=f"{train_confusion_loss / (i + 1):.4f}",
            )

            # scheduler_enc.step()
            # scheduler_dec.step()
            # scheduler_disc.step()

        train_recon_loss /= len(train_loader)
        train_disc_loss /= len(train_loader)
        train_confusion_loss /= len(train_loader)

        print(
            f"Epoch [{e + 1}/{args.epochs}], Recon Loss: {train_recon_loss:.4f}, Disc Loss: {train_disc_loss:.4f}, Confusion Loss: {train_confusion_loss:.4f}"
        )

        # Save Intermediary Models:
        save_model(
            encoder,
            enc_optim,
            e,
            args=args,
            path=f"models/aae/{args.dataset}/epoch_{e}/Encoder",
        )
        save_model(
            decoder,
            dec_optim,
            e,
            args=args,
            path=f"models/aae/{args.dataset}/epoch_{e}/Decoder",
        )
        save_model(
            discriminator,
            dec_optim,
            e,
            args=args,
            path=f"models/aae/{args.dataset}/epoch_{e}/Discriminator",
        )


if __name__ == "__main__":
    args = load_train_args()
    if args.adversarial:
        train_with_discriminator(args)
    else:
        train_reconstruction(args)
