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
            nn.Conv3d(n_channels, ndf, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf, ndf * 2, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 2, ndf * 4, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, ndf * 8, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # input shape: (1, 3, d, h, w)
        start = 0
        end = 96

        out = []

        while end < input.shape[2]:
            x = input[:,:,start:end,:,:]
            # x.shape: 1, 3, 96, h, w
            h = self.main(x).squeeze()
            out.append(h)
            start += 6
            end += 6

        out = torch.cat(out, dim=0)

        return out, None