import torch
import torch.nn as nn
import numpy as np


class MNISTencoder(nn.Module):
    def __init__(self, args):
        super(MNISTencoder, self).__init__()
        self.shape = (1, 28, 28)
        self.dim = args.dim

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
            nn.Dropout(args.dropout),
            nn.ReLU(True),
            nn.Conv2d(self.dim, 2 * self.dim, 5, stride=2, padding=2),
            nn.Dropout(args.dropout),
            nn.ReLU(True),
            nn.Conv2d(2 * self.dim, 4 * self.dim, 5, stride=2, padding=2),
            nn.Dropout(args.dropout),
            nn.ReLU(True),
        )
        self.output = nn.Linear(4 * 4 * 4 * self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.conv_block(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)  # Flatten
        out = self.output(out)
        return out.view(-1, self.dim)


class MNISTdecoder(nn.Module):
    def __init__(self, args):
        super(MNISTdecoder, self).__init__()
        self.dim = args.dim

        self.preprocess = nn.Sequential(
            nn.Linear(self.dim, 4 * 4 * 4 * self.dim),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                4 * self.dim,
                2 * self.dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.dim,
                self.dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(
            self.dim,
            1,
            kernel_size=2,
            stride=2,
            padding=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.preprocess(input)
        out = out.view(-1, 4 * self.dim, 4, 4)
        out = self.block1(out)
        out = self.block2(out)
        out = self.deconv_out(out)
        out = self.sigmoid(out)
        return out.view(-1, 784)
