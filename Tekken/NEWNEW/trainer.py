import time
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import vis_tool
from config import get_config
import os
from cnn_extractor import CNN, GRU


class Trainer(object):
    def __init__(self, config, h_loader, r_loader, t_loader):
        self.config = config
        self.h_loader = h_loader
        self.r_loader = r_loader
        self.t_loader = t_loader

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


        for l,p in self.c2d.named_parameters():

            p.requires_grad = False


        self.gru = GRU(self.c2d).cuda()


    def train(self):
        # create optimizers
        cfig = get_config()
        opt = optim.RMSprop(filter(lambda p: p.requires_grad, self.gru.parameters()),
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        start_time = time.time()



        criterion = nn.BCELoss()

        max_acc = 0.

        for epoch in range(self.n_epochs):
            epoch_loss = []
            for step, (h, r) in enumerate(zip(self.h_loader, self.r_loader)):
                self.gru.train()

                #if step == 3: break
                h_video = h
                r_video = r

                # self.vis.img("h",h_video)
                # self.vis.img("r", r_video)

                # highlight video
                h_video = Variable(h_video).cuda()
                r_video = Variable(r_video).cuda()


                self.gru.zero_grad()

                h_loss = self.gru(h_video)

                # target = torch.ones(predicted.shape, dtype=torch.float32).cuda()
                #
                # h_loss = criterion(predicted, target)  # compute loss

                # for n, p in self.gru.named_parameters():
                #     if "c2d" not in n:
                #         print(n)
                #         print(p)

                h_loss.backward()
                opt.step()

                # for n, p in self.gru.named_parameters():
                #     if "c2d" not in n:
                #         print(n)
                #         print(p)

                self.gru.zero_grad()



                step_end_time = time.time()

                epoch_loss.append((h_loss.data).cpu().numpy())

                print('[%d/%d][%d/%d] - time: %.2f, h_loss: %.3f'
                      % (epoch + 1, self.n_epochs, step + 1, self.n_steps, step_end_time - start_time,
                         h_loss
                         ))

                self.vis.plot('H_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                              (h_loss.data).cpu().numpy())

                # self.vis.plot('R_LOSS with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f'
                #               % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                #               (r_loss.data).cpu().numpy())

            self.vis.plot("Avg loss plot with lr:%.4f, b1:%.1f, b2:%.3f, wd:%.5f"
                              % (cfig.lr, cfig.beta1, cfig.beta2, cfig.weight_decay),
                          np.mean(epoch_loss))

            if epoch % self.checkpoint_step == 0:
                accuracy,savelist = self.test(self.t_loader)

                if accuracy > max_acc:
                    max_acc = accuracy
                    torch.save(self.gru.state_dict(), './samples/lr_%.4f_chkpoint'%cfig.lr + str(epoch + 1) + '.pth')
                    for f in savelist:
                        np.save("./samples/"+f[0]+".npy",f[1])
                    print(np.load("./samples/testRV04(198,360).npy"))
                    print("checkpoint saved")


    def test(self, t_loader):
        self.gru.eval()
        accuracy = 0.

        savelist = []

        total_len = len(t_loader)

        for step, (tv, label, filename) in enumerate(t_loader):
            filename = filename[0].split(".")[0]

            label = label.squeeze()

            start = 0
            end = 24

            correct = 0
            count = 0

            npy = np.zeros(tv.shape[1])

            while end < tv.shape[1]:

                t_video = Variable(tv[:,start:end,:,:,:]).cuda()
                predicted = self.gru(t_video)

                gt_label = label[start:end]

                if len(gt_label[gt_label==1.]) > 12:
                    gt_label = torch.ones(predicted.shape,dtype=torch.float32).cuda()

                else: gt_label = torch.zeros(predicted.shape,dtype = torch.float32).cuda()


                if predicted < 0.5:
                    npy[start:end] = 1.


                predicted[predicted<0.5] = 1.
                predicted[predicted>=0.5] = 0.


                correct += (predicted == gt_label).item()

                start += 24
                end += 24
                count += 1

            accuracy += (correct / count)/total_len

            savelist.append([filename,npy])



        print("Accuracy:",round(accuracy,4))
        self.vis.plot("Accuracy with lr:%.3f"%self.lr,accuracy)

        return accuracy, savelist
