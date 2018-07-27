"""
Usage: main.py [options] --dataroot <dataroot> --cuda
"""

import os

import random
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from trainer import Trainer

from dataloader import get_loader


def main(config):
    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataroot = config.dataroot
    h_datapath = os.path.join(dataroot, "HV")
    r_datapath = os.path.join(dataroot, "RV")
    t_datapath = os.path.join(dataroot, 'testRV')

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    h_loader, r_loader, t_loader = get_loader(h_datapath, r_datapath, t_datapath)
    config.n_steps = min(len(h_loader), len(r_loader))

    trainer = Trainer(config, h_loader, r_loader, t_loader)
    trainer.train()
    # trainer.save_npy(t_loader)

if __name__ == "__main__":
    config = get_config()
    main(config)
