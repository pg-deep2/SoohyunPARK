import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class Discriminator(nn.Module):
    def __init__(self, n_channels=3, n_output_neurons=1, ndf=16):
        super(Discriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons


        self.main = nn.Sequential(
            nn.Conv3d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None