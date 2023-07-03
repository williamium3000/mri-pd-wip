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


class PDWIP2DDataset(Dataset):
    def __init__(self, cfg, mode, pd_root, wip_root, list):
        self.mode = mode # train or val
        self.pd_root = pd_root
        self.wip_root = wip_root
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
                transforms.ToTensor(),
            ])
        elif mode == "val":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        

    def __getitem__(self, item):
        pd_path = self.pd_images[item]
        wip_path = self.wip_images[item]
        
        pd_img = np.load(pd_path) # (X, Y, C)
        wip_img = np.load(wip_path) # (X, Y, C)

        pd_img = self.transform(pd_img).float()
        wip_img = self.transform(wip_img).float()
        return pd_img, wip_img

    def __len__(self):
        return len(self.pd_images)
