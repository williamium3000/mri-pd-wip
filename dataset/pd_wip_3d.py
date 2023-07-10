from dataset.transform3d import *

from copy import deepcopy
import math
import numpy as np
import os
import random
import nibabel as nib

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PDWIP3DDataset(Dataset):
    def __init__(self, cfg, mode, pd_root, wip_root, list, return_name):
        self.mode = mode # train or val
        self.pd_root = pd_root
        self.wip_root = wip_root
        self.return_name = return_name
        self.list = list
        with open(list, "r") as f:
            filenames = f.readlines()
        pd_images = []
        wip_images = []
        for line in filenames:
            pd_file, wip_file = line.strip().split(" ")
            pd_images.append(os.path.join(pd_root, pd_file))
            wip_images.append(os.path.join(wip_root, wip_file))
        self.pd_images = pd_images
        self.wip_images = wip_images
        
        if mode == "train":
            self.transform = transforms.Compose([
                NoneZeroRegion3D(),
                RandomCrop3D(cfg["crop_size"]),
                RandomFlip3D(prob=0.5),
                transforms.RandomChoice([
                    RandomRotation90n3d(prob=1.0),
                    RandomRotation3d(prob=1.0),
                    Identity()
                ]),
                StdNormalize(),
                ToTensor3d(),
            ])
        elif mode == "val":
            self.transform = transforms.Compose([
                NoneZeroRegion3D(),
                StdNormalize(),
                Pad3D(size_divisor=16),
                ToTensor3d(),
            ])
        

    def __getitem__(self, item):
        pd_path = self.pd_images[item]
        wip_path = self.wip_images[item]
        # X, Z, Y
        pd_img = nib.load(pd_path).get_fdata().transpose(0,2,1)[:, :, :, np.newaxis] # (X, Y, Z, C)
        wip_img = nib.load(wip_path).get_fdata().transpose(0,2,1)[:, :, :, np.newaxis]

        # pd_img = self.transform(pd_img).float()
        # wip_img = self.transform(wip_img).float()
        # pd max:  1762.0
        # wip max:  1848.0
        pd_img = torch.tensor(pd_img).permute(3, 0, 1, 2).float() / 1762.0
        wip_img = torch.tensor(wip_img).permute(3, 0, 1, 2).float() / 1848.0
        if self.return_name:
            return pd_img, wip_img, os.path.basename(wip_path)
        else:
            return pd_img, wip_img

    def __len__(self):
        return len(self.pd_images)

