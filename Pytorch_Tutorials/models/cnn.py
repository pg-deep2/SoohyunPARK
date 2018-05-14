import torch
import torch.nn as nn
import torch.nn.parallel


######################################################
############ Design Your OWN model HERE! #############
######################################################

class _CNN(nn.Module):
    def __init__(self, ngpu, nc, nclass):
        super(_CNN, self).__init__()
        self.ngpu = ngpu

        # input size. bs(batch size), nc, 64 (image size), 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, 16, 4, 2, 1),
            # state size. bs, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # state size. bs, 16, 16, 16
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1),
            # state size. bs, 32, 8, 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # state size. bs, 32, 4, 4
        )

        self.fc = nn.Linear(4 * 4 * 32, nclass)


    def main(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward(self, input):
        # Only proceeds layers called in this method!
        # forward() should be overridden by you!
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


