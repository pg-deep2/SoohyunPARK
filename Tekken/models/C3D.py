import torch.nn as nn
import torch
from dataloader import get_loader

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    # def forward(self, x):
    #     print("x",x.shape)
    #     x = x.permute(0, 2, 1, 3, 4)
    #
    #     print(x.shape)
    #
    #
    #     #print(x.shape)
    #
    #     h = self.relu(self.conv1(x))
    #     h = self.pool1(h)
    #
    #     h = self.relu(self.conv2(h))
    #     h = self.pool2(h)
    #
    #     h = self.relu(self.conv3a(h))
    #     h = self.relu(self.conv3b(h))
    #     h = self.pool3(h)
    #
    #     h = self.relu(self.conv4a(h))
    #     h = self.relu(self.conv4b(h))
    #     h = self.pool4(h)
    #
    #     h = self.relu(self.conv5a(h))
    #     h = self.relu(self.conv5b(h))
    #     h = self.pool5(h)
    #
    #
    #     h = h.view(-1, 8192)
    #     print()
    #     h = self.relu(self.fc6(h))
    #     h = self.dropout(h)
    #     h = self.relu(self.fc7(h))
    #     h = self.dropout(h)
    #
    #     logits = self.fc8(h)
    #     probs = self.softmax(logits)
    #
    #     return probs


class GRU(nn.Module):
    def __init__(self, c3d):
        super(GRU, self).__init__()

        self.c3d = c3d

        # 2048 -> output size[1] of resnet
        self.gru = nn.GRUCell(27, 10)
        self.relu = nn.ReLU()
        self.ae = nn.Sequential(nn.Linear(5120,100),
                                nn.ReLU(),
                                nn.Linear(100,5120),
                                nn.Tanh())

        self.mseLoss = nn.MSELoss()


    def forward(self, input):
        print("input",input.shape)
        start = 0
        end = 48

        loss_list = []

        e_t = torch.FloatTensor(512, 10).normal_().cuda()

        while end < input.shape[2]:
            x = input[:, :, start:end, :, :]
            # x.shape: 1, 3, 96, h, w
            h = self.c3d(x)
            h = h.squeeze()
            h = h.view(512,-1)
            # h.shape: 1,512, 27

            print("h",h.shape)

            e_t = (self.gru(h.cuda(), e_t))

            print("et",e_t.shape)
            # e_t.shape: 512,128

            feature_in = e_t.contiguous().view(1,-1)

            feature_out = self.ae(feature_in)
            print("fo",feature_out.shape)

            loss = torch.mean(torch.abs(feature_out - feature_in))
            loss_list.append(loss.data)

            start += 6
            end += 6

        total_loss = sum(loss_list)

        return total_loss
