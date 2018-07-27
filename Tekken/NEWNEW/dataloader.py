from __future__ import print_function
import os, random
import cv2
import numpy as np
import torch
from torchvision import transforms
import functools
import torch.utils.data
import visdom, time
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#

class HighlightVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot, v) for v in videofiles if os.path.splitext(v)[-1] == '.mp4']
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        frames = []
        label = 1

        # reading video using cv2
        while True:
            ret, frame = cap.read()
            if ret:
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)

                # transform. 270x480 -> 240x400
                frames.append(frame)
                # transform. normalize

                # print(frame)
            else:
                break
        cap.release()

        out = np.concatenate(frames)
        out = out.reshape(-1,3,270,480)

        return self.transforms(out)

    def __len__(self):
        return len(self.videofiles)


class RawVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot, v) for v in videofiles if os.path.splitext(v)[-1] == '.mp4']
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        f = 0
        frames = []
        label = 0

        # reading video using cv2
        while True:
            ret, frame = cap.read()
            if ret:
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                # transform. 270x480 -> 240x400
                frames.append(frame)
                f += 1
                # print(frame)
            else:
                break
        cap.release()

        out = np.concatenate(frames)
        out = out.reshape(-1,3,270,480)


        return self.transforms(out)

    def __len__(self):
        return len(self.videofiles)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot, v) for v in videofiles if os.path.splitext(v)[-1] == '.mp4']
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        frames = []

        filename = os.path.split(self.videofiles[item])[-1]
        h_start = filename.index("(")
        h_end = filename.index(")")
        h_frames = filename[h_start + 1: h_end]

        # reading video using cv2
        while True:
            ret, frame = cap.read()
            if ret:
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                # transform. 270x480 -> 240x400
                # frame = frame[:, 15:255, 40:440]
                frames.append(frame)
                # print(frame)
            else:
                break
        cap.release()

        out = np.concatenate(frames)
        out = out.reshape(-1,3,270,480)

        label = np.zeros(out.shape[0])

        if "," in h_frames:
            s, e = h_frames.split(',')
            label[int(s):int(e)] = 1.

        return self.transforms(out), label, filename

    def __len__(self):
        return len(self.videofiles)


def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for im in video:
        vid.append(image_transform(im.transpose(1, 2, 0)))

    vid = torch.stack(vid)
    # vid. 10, 3, 240, 260
    return vid


def get_loader(h_dataroot, r_dataroot, t_dataroot, batch_size=1):
    image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.Resize((350,350)),
        transforms.RandomCrop(299,),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    h_dataset = HighlightVideoDataset(h_dataroot, video_transforms)
    r_dataset = RawVideoDataset(r_dataroot, video_transforms)
    t_dataset = TestDataset(t_dataroot, video_transforms)

    h_loader = DataLoader(h_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    r_loader = DataLoader(r_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    t_loader = DataLoader(t_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    return h_loader, r_loader, t_loader


if __name__ == "__main__":
    h, r, t = get_loader(r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\HV",
                         r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\RV",
                         r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ve r2\testRV", 1)

    print(h)
    print(type(h))
    for a, b in enumerate(h):
        print(a,b[0].shape)
#
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
#
#
#
# def get_loader(h_dataroot, r_dataroot, t_dataroot, batch_size=20):
#     image_transforms = transforms.Compose([
#         # Image.fromarray,
#         transforms.Resize((299,299)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     # video_transforms = functools.partial(video_transform, image_transform=image_transforms)
#
#     dataroot = "C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\Images\Train"
#
#     r_dset = D(dataroot + "\RV", image_transforms)
#     h_dset = D(dataroot + "\HV", image_transforms)
#
#     dataroot = "C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\Images\Test"
#     t_r_dset = D(dataroot + "\RV_filtered", image_transforms)
#     # t_h_dset = D(dataroot + "\HV_filtered", image_transforms)
#
#     h_loader = DataLoader(h_dset, batch_size=batch_size, drop_last=True, shuffle=True)
#     r_loader = DataLoader(r_dset, batch_size=batch_size, drop_last=True, shuffle=True)
#     t_loader = DataLoader(t_r_dset, batch_size=batch_size, drop_last=True, shuffle=False)
#
#     return h_loader, r_loader, t_loader