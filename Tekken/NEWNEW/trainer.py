import time
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import vis_tool
from config import get_config

from cnn_extractor import CNN, GRU


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
        self.n_steps = config.n_steps
        self.log_interval = int(config.log_interval)  # in case
        self.checkpoint_step = int(config.checkpoint_step)

        self.use_cuda = config.cuda
        self.outf = config.outf
        self.build_model()
        self.vis = vis_tool.Visualizer()

    def build_model(self):
        self.c2d = CNN().cuda()
        self.c2d.load_state_dict(torch.load('cnn.pkl'))  # load pre-trained cnn extractor


        # for l,p in self.c2d.named_parameters():
        #     print(l)

        # self.c2d = nn.Sequential(*list(self.c2d.children())[:-1])
        for l,p in self.c2d.named_parameters():

            p.requires_grad = False
            # print(l,p.requires_grad)

        # c2d_layer = list(self.c2d.children())
        # fixed_layers = []
        # 
        # for param in self.c2d.parameters():
        #     param.requires_grad = False
        #     print(param.requires_grad)
            # no trainable parameters
            # fixed_layers.append(layer)
        # 
        # self.c2d = nn.Sequential(*fixed_layers).cuda()
        
        # 여기서 c2d fix 시키게 어떻게 함..? ㅠㅠㅠㅠㅠ

        self.gru = GRU(self.c2d).cuda()
        # for l,p in self.c2d.named_parameters():
        #     if "c2d" in l:
        #         p.requires_grad = False
        #         print(l,p.requires_grad)
        #     else:
        #         # p.requires_grad = True
        #         print(l)


    def train(self):
        # create optimizers
        cfig = get_config()
        opt = optim.RMSprop(filter(lambda p: p.requires_grad, self.gru.parameters()),
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        start_time = time.time()

        self.gru.train()

        criterion = nn.BCELoss()

        print(list(self.gru.named_parameters()))

        for epoch in range(self.n_epochs):
            epoch_loss = []
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                h_video = h
                r_video = r
                self.vis.img("h",h_video)
                self.vis.img("r", r_video)

                # highlight video
                h_video = Variable(h_video.cuda())
                r_video = Variable(r_video.cuda())

                self.gru.zero_grad()

                predicted = self.gru(h_video.cuda())  # predicted snippet's score
                # print("Predicted:", predicted)
                # print("Predicted shape:", predicted.shape)
                #print(predicted)
                target = Variable(torch.ones(predicted.shape, dtype=torch.float32).cuda())
                # print("Target:", target)
                # print("Target shape:", target.shape)
                # target = torch.from_numpy(
                #    np.ones([len(predicted)], dtype=np.float)).cuda()  # highlight videos => target:1
                h_loss = Variable(criterion(predicted, target).cuda(), requires_grad=True)  # compute loss

                for n, p in self.gru.named_parameters():
                    if 'c2d' not in n:
                        print(n)
                        print(p)
                    # print(n,p.requires_grad)
                print()
                print()
                h_loss.backward()
                opt.step()

                for n, p in self.gru.named_parameters():
                    if 'c2d' not in n:
                        print(p)
                    # print(n,p.requires_grad)
                print()
                print()


                self.gru.zero_grad()

                predicted = self.gru(r_video.cuda())  # predicted snippet's score

                target = Variable(torch.zeros(predicted.shape, dtype=torch.float32).cuda())
                r_loss = Variable(criterion(predicted, target), requires_grad=True)  # compute loss

                r_loss.backward()
                opt.step()

                step_end_time = time.time()

                total_loss = r_loss + h_loss
                epoch_loss.append((total_loss.data).cpu().numpy())

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f, r_loss: %.3f, total_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, self.n_steps, step_end_time - start_time, h_loss, r_loss,
                         total_loss))

                self.vis.plot('H_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())

                self.vis.plot('R_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (r_loss.data).cpu().numpy())

            self.vis.plot("Avg loss plot", np.mean(epoch_loss))

            if epoch % self.checkpoint_step == 0:
                torch.save(self.gru.state_dict(), 'chkpoint' + str(epoch + 1) + '.pth')
                print("checkpoint saved")
