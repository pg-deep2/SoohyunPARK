import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models import model

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Trainer(object):
    def __init__(self, config, h_loader, r_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader

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
        print("[*] Load models from {}...".format(self.outf))


    def build_model(self):
        self.model = model.Discriminator().cuda()


        if self.outf != None:
            self.load_model()


    def train(self):
        self.loss = nn.BCELoss()

        # if self.use_cuda:
        #     self.model.cuda()

        # create optimizers
        opt_model = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                   weight_decay=self.weight_decay)

        start_time = time.time()

        for epoch in range(self.n_epochs):

            for step, (h_video, h_label) in enumerate(self.h_loader):
                h_video = Variable(h_video.cuda())
                h_video = h_video.permute(0,2,1,3,4)
                if h_video.shape[2] < 96: continue

                self.model.zero_grad()

                output, fake_categ = self.model(h_video)
                loss = self.loss(output+1e-10, Variable(torch.ones(output.size()).cuda()))

                loss.backward()
                opt_model.step()

                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time: %.2f, loss: %.3f'
                      % (epoch, self.n_epochs, step, len(self.h_loader), step_end_time - start_time,
                         loss))

                if step % self.log_interval == 0:
                    pass

            if epoch % self.checkpoint_step == 0:
                pass

    def _get_variable(self, inputs):
        out = Variable(inputs.cuda())
        return out
