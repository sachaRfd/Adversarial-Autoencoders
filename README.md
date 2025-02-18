# Adversarial Autoencoders

A generative model using adversarial autoencoders for MNIST and timeseries data.

## Installation

Create and activate a conda environment:

```bash
conda create -n aaa python=3.12 -y
conda activate aaa
```

Then install the local package:

```bash
pip install -e .
```

## Dataset

Supported datasets:

- MNIST


## Training

1. To train a simple autoencoder with reconstruction loss:

   ```bash
   python -m train
   ```

2. To train with adversarial training enabled:

   ```bash
   python -m train --adversarial True
   ```

## Sampling

To generate and visualize samples, run the following command. This will output a set of reconstructed images along with samples generated from a Normally distributed latent space.

```bash
python -m sample --model_path PATH/TO/MODEL-WEIGHTS
```

## About Adversarial Autoencoders

Adversarial Autoencoders (AAEs) combine autoencoding with adversarial training to enforce a structured latent space.

## Method

An Adversarial Autoencoder operates in three key steps:

1. **Compression**: The encoder compresses the input data into a latent representation, capturing its essential features.
2. **Reconstruction**: The decoder reconstructs the original input from the latent representation, ensuring that encoded data retains critical information.
3. **Adversarial Training**: A discriminator ensures that the latent space follows a chosen prior distribution. The encoder learns to generate latent codes indistinguishable from this prior, leading to a more structured and robust representation.

## Differences Compared to GANs

While both Adversarial Autoencoders (AAEs) and Generative Adversarial Networks (GANs) use adversarial training, they differ in their approach and objectives:

- **Latent Space Regularization**: AAEs explicitly enforce a prior distribution on the latent space using a discriminator, whereas GANs do not have a predefined latent space structure.
- **Reconstruction Objective**: AAEs incorporate a reconstruction loss to ensure meaningful latent representations, while GANs focus solely on generating realistic outputs without an explicit reconstruction constraint.
- **Training Stability**: AAEs tend to be more stable during training compared to GANs, as they leverage autoencoder training alongside adversarial objectives.
- **Applications**: AAEs are well-suited for representation learning and semi-supervised learning, whereas GANs are primarily used for image synthesis and other generative tasks.

