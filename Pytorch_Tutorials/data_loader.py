import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_loader(dataroot, batch_size, num_workers, image_size, shuffle=True):
    train_path = os.path.join(dataroot, "train/")
    tr_dataset = dset.ImageFolder(root=train_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))


    assert tr_dataset
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)


    test_path = os.path.join(dataroot, "test/")
    ts_dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    assert ts_dataset
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return tr_dataloader, ts_dataloader
