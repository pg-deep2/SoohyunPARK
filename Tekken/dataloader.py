from __future__ import print_function
import os, random
import cv2
import numpy as np
import torch, time
import visdom
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import functools

class HighlightVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot,v) for v in videofiles if os.path.splitext(v)[-1]=='.mp4']
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        frames = []
        label = self.videofiles[item].split("\\")[-2]

        # reading video using cv2
        while True:
            ret, frame = cap.read()
            if ret:
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
                # print(frame)
            else:
                break

        out = np.concatenate(frames)
        out = out.reshape(-1, 3, 270, 480)

        return self.transforms(out)

    def __len__(self):
        return len(self.videofiles)


class RawVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot,v) for v in videofiles if os.path.splitext(v)[-1]=='.mp4']
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        frames = []
        label = self.videofiles[item].split("\\")[-2]

        # reading video using cv2
        f = 0
        while True:
            ret, frame = cap.read()
            if ret:
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
                # print(frame)
                f += 1
            else:
                break

        total_frames = f
        # choose length of raw video [6,10]
        if total_frames >= 12*10:
            frame_len = random.randint(6,10)*12

        # if video is shorter than 120 frames, [6, secs]
        elif total_frames > 12 * 6:
            frame_len = random.randint(6,total_frames//12)*12

        else:
            raise IndexError('Video is shorter than 6 seconds!')

        # start frame: [0, (total frame - frame len))
        random_start = random.randint(0,total_frames - frame_len)
        out = np.concatenate(frames)
        out = out.reshape(-1, 3, 270, 480)
        out = out[random_start : random_start+frame_len, : , : , :]

        return self.transforms(out)

    def __len__(self):
        return len(self.videofiles)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videofiles = os.listdir(dataroot)
        self.videofiles = [os.path.join(self.dataroot,v) for v in videofiles if os.path.splitext(v)[-1]=='.mp4']
        self.textfiles = [os.path.join(self.dataroot,v) for v in videofiles if os.path.splitext(v)[-1]=='.txt']

    def __getitem__(self, item):
        cap = cv2.VideoCapture(self.videofiles[item])
        frames = []

        filename = os.path.split(self.videofiles[item])[-1]
        h_start = filename.index("(")
        h_end = filename.index(")")
        h_frames = filename[h_start+1 : h_end]

        # reading video using cv2
        while True:
            ret, frame = cap.read()
            if ret:
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
                # print(frame)
            else:
                break

        out = np.concatenate(frames)
        out = out.reshape(-1, 3, 270, 480)

        label = np.zeros(out.shape[0])

        if "," in h_frames:
            s,e = h_frames.split(',')
            label[int(s):int(e)] = 1.

        return self.transforms(out), label

    def __len__(self):
        return len(self.videofiles)

def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for im in video:
        vid.append(image_transform(im.transpose(1,2,0)))

    vid = torch.stack(vid)
    # vid. 10, 3, 64, 64
    vid = vid.permute(1, 0, 2, 3)
    # vid. 3, 10, 64, 64
    return vid

def get_loader(h_dataroot, r_dataroot, test_dataroot, batch_size=1):
    image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.CenterCrop(270),
        transforms.ToTensor(),
        # lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    h_dataset = HighlightVideoDataset(h_dataroot, video_transforms)
    # h_video = h_dataset[2][0]

    # viz = visdom.Visdom()
    # for f in range(0, h_video.shape[0]):
    #     viz.image(h_video[f,:,:,:], win="gt video", opts={'title':'GT'})
    #     time.sleep(0.01)


    r_dataset = RawVideoDataset(r_dataroot, video_transforms)
    # r_video = r_dataset[-1][0]

    # viz = visdom.Visdom()
    # for f in range(0, r_video.shape[0]):
    #     viz.image(r_video[f,:,:,:], win="random video", opts={'title':'RANDOM'})
    #     time.sleep(0.01)
    test_dataset = TestDataset(test_dataroot, video_transforms)

    h_loader = DataLoader(h_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    r_loader = DataLoader(r_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    return h_loader, r_loader, test_loader
if __name__=="__main__":
    get_loader("C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\HV",
               "C:\\Users\msi\Desktop\Soohyun\프로그라피\\2기_디비디비딥\VideoData\RV", 1)
