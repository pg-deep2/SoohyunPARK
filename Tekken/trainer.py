import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models.C3D import C3D, GRU

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Trainer(object):
    def __init__(self, config, h_loader, r_loader, test_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader
        self.test_loader = test_loader

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.n_epochs = config.n_epochs
        self.log_interval = int(config.log_interval)
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda

        self.outf = config.outf

        self.build_model()

        # if self.use_cuda:
        #     self.model.cuda()

    def load_model(self):
        self.p3d.load_state_dict(torch.load(self.config.pretrained_path))
        # p3d_net.cuda()
        # print("before fine tuning:", p3d_net)

        # FC layer removal & fixing pretrained layers' parameter
        fc_removed = list(self.p3d.children())[:-6]

        _p3d_net = []
        relu = nn.ReLU()

        for layer in fc_removed:
            for param in layer.parameters():
                param.requires_grad = False
            if layer.__class__.__name__ == 'MaxPool3d':
                _p3d_net.extend([layer, relu])
            else:
                _p3d_net.append(layer)

        # last_conv3d = nn.Conv3d(512, 512, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
        # _p3d_net.extend([last_conv3d, relu])

        p3d_net = nn.Sequential(*_p3d_net).cuda()

        self.p3d = p3d_net


    def build_model(self):
        self.p3d = C3D().cuda()
        self.load_model()

        self.gru = GRU(self.p3d).cuda()
        print("MODEL:")
        print(self.gru)

    def train(self):
        # if self.use_cuda:
        #     self.model.cuda()

        # create optimizers
        opt_model = optim.Adam(filter(lambda p: p.requires_grad,self.gru.parameters()),
                               lr=self.lr, betas=(self.beta1, self.beta2),
                               weight_decay=self.weight_decay)

        start_time = time.time()

        self.gru.train()

        for epoch in range(self.n_epochs):
            # common_len = min(len(self.h_loader),len(self.r_loader))
            for step, (h, r) in enumerate(zip(self.h_loader,self.r_loader)):
                h_video = h
                r_video = r
                # highlight video
                h_video = Variable(h_video.cuda())

                self.gru.zero_grad()

                h_loss = Variable(self.gru(h_video).cuda(),requires_grad=True)

                h_loss.backward()
                opt_model.step()

                # raw video
                r_video = Variable(r_video.cuda())
                self.gru.zero_grad()

                r_loss = Variable(self.gru(r_video).cuda(),requires_grad=True)

                r_loss.backward()
                opt_model.step()

                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f'
                      % (epoch+1, self.n_epochs, step+1, min(len(self.h_loader),len(self.r_loader)),
                         step_end_time - start_time, h_loss, r_loss))

                if step % self.log_interval == 0:
                    # for step, t in enumerate(self.test_loader):
                    #     t_video = t[0]
                    #     t_label = t[1]
                    #
                    #
                    pass

            if epoch % self.checkpoint_step == 0:
                pass

