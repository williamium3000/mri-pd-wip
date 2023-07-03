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


class Medical3DDataset(Dataset):
    def __init__(self, cfg, mode, img_root, label_root, list):
        self.mode = mode # train or val
        self.img_root = img_root
        self.label_root = label_root
        self.list = list
        with open(list, "r") as f:
            filenames = f.readlines()
        images = []
        for filename in filenames:
            images.append(
                os.path.join(img_root, filename.strip() + ".nii.gz")
            )
        self.images = images
        labels = []
        for filename in filenames:
            labels.append(
                os.path.join(label_root, filename.strip() + ".nii.gz")
            )
        self.labels = labels
        
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
        img_path = self.images[item]
        mask_path = self.labels[item]
        img = nib.load(img_path).get_fdata() # (X, Y, Z, C)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        
        img, mask = self.transform((img, mask))
        

        return img, mask

    def __len__(self):
        return len(self.images)
