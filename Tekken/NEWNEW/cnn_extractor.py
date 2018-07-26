import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import os

import numpy as np

# Hyper Parameters
num_epochs = 100
batch_size = 20
learning_rate = 0.0001


# class D(Dataset):
#     def __init__(self, root_dir, transform=None):
#
#         self.dataroot = root_dir
#         self.im_files = os.listdir(self.dataroot)
#
#         self.transforms = transform if transform is not None else lambda x: x
#
#     def __len__(self):
#         return len(self.im_files)
#
#     def __getitem__(self, idx):
#         im = Image.open(os.path.join(self.dataroot,self.im_files[idx]))
#
#
#         return self.transforms(im)
#
# image_transforms = transforms.Compose([
#         transforms.Resize((299,299)),
#         transforms.ToTensor(),
#         # lambda x: x[:n_channels, ::],
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#
# dataroot = "C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\Images\Train"
#
# r_dset = D(dataroot+"\RV",image_transforms)
# h_dset = D(dataroot+"\HV",image_transforms)
#
# dataroot = "C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\Images\Test"
# t_r_dset = D(dataroot+"\RV_filtered",image_transforms)
# t_h_dset = D(dataroot+"\HV_filtered",image_transforms)
#
#
# # Data Loader (Input Pipeline)
# r_l = torch.utils.data.DataLoader(dataset=r_dset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
# h_l = torch.utils.data.DataLoader(dataset=h_dset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# t_r_l = torch.utils.data.DataLoader(dataset=t_r_dset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
#
# t_h_l = torch.utils.data.DataLoader(dataset=t_h_dset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# CNN Model (2 conv layer)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, 4, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )
        self.fc = nn.Conv2d(256,1,1,1,0)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # 24, 256, 1, 1
        # print(out.shape)
        # out = self.fc(out)
        # out = self.sig(out)
        return out

class GRU(nn.Module):
    def __init__(self, cnn):
        super(GRU, self).__init__()

        self.c2d = cnn
        self.gru1 = nn.LSTM(256, 16, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(16,1),
                                nn.Sigmoid())


        # self.relu = nn.ReLU()
        # self.gru2 = nn.GRU(10,1)


        #self.sig = nn.Sigmoid()

    def forward(self, input):
        input = input.view(-1,3,299,299)
        h = self.c2d(input)
        h = h.view(1,-1,256)
        # print(h.shape)
        h,_ = self.gru1(h)
        # print(h.shape) # 1, 16
        h = nn.ReLU()(h)
        # print(h.shape)
        h = h.view(-1,16)
        h = self.fc(h)

        return h





        # f_score_list = np.zeros(input.shape[1])
        # step = 0
        # start = 0
        # end = 24
        # while end < input.shape[1]:
        #     x = input[0, start:end, :, :, :]
        #     # print(x.shape)
        #     x = x.squeeze()
        #
        #     h = self.c2d(x)
        #     # print(h.shape)
        #     h = h.squeeze()
        #     #print('c2d output shape:', h.shape) # [24, 256]
        #     h = h.view(-1, 24, 256)
        #     # print(h)
        #     # print("h", h.shape)
        #     # h.shape: 48 2048 2 2
        #
        #     h,_  = self.gru1(h)
        #     # h = self.relu(h)
        #     # h,_ = self.gru2(h)
        #     # h = self.fc(h)
        #     # print("h", h.shape, h)
        #     #h_out = self.sig(h)
        #     h_out = h.squeeze()
        #     # print("h", h.shape, h)
        #
        #     # h.shape : 1
        #     #print("h_out:", h_out)
        #     # ??? 어캐해야하징~~
        #     f_score_list[start:end] += h_out.data
        #     # print(f_score_list)
        #
        #     start += 6
        #     end += 6
        #     step += 1
        #
        # f_score_list[6:12] /= 2
        # f_score_list[12:18] /= 3
        # f_score_list[18:start] /= 4
        # f_score_list[start:start + 6] /= 3
        # f_score_list[start + 12:start + 18] /= 2
        #
        # f_score_list = f_score_list[:end - 6]
        # return torch.from_numpy(f_score_list).cuda()
        
# cnn = CNN()
# cnn.cuda()
#
# # Loss and Optimizer
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#
#
# print("start training")
#
# # Train the Model
# for epoch in range(num_epochs):
#     for i, (r, h) in enumerate(zip(r_l,h_l)):
#         images = Variable(r).cuda()
#         # labels = Variable(labels).cuda()
#         labels = torch.zeros(images.shape[0]).cuda()
#         # print(images)
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = cnn(images)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#
#         images = Variable(h).cuda()
#
#         labels = torch.ones(images.shape[0]).cuda()
#
#         # print(images)
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = cnn(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 1 == 0:
#             print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
#                    %(epoch+1, num_epochs, i+1, min(len(r_l),len(h_l)), loss.data[0]))
#
# # Test the Model
#     cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
#     correct = 0
#     total = 0
#     for r, v in zip(t_r_l, t_h_l):
#
#         images = torch.stack((r,v)).view(-1,3,299,299).cuda()
#
#         r_label = torch.zeros(images.shape[0]//2)
#         h_label = torch.ones(images.shape[0]//2)
#
#         labels = torch.stack((r_label,h_label)).view(images.shape[0]).cuda()
#
#
#         outputs = cnn(images)
#         outputs = outputs.squeeze()
#         outputs[outputs > 0.7] = 1.0
#         outputs[outputs < 0.3] = 0.0
#         print(outputs)
#         total += labels.size(0)
#         correct += (outputs == labels).sum()
#
#     print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
#
# print(outputs)
# # # Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pkl')