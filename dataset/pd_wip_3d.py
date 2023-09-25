from dataset.paired_transform3d import *

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
    def __init__(self, cfg, mode, pd_root, wip_root, list, return_name=False):
        self.mode = mode # train or val
        self.pd_root = pd_root
        self.wip_root = wip_root
        self.return_name = return_name
        self.list = list
        # self.norm_type = cfg["norm_type"]
        # self.quantile_clip = cfg["quantile_clip"]
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
            if cfg.get("use_augmentation", True):
                self.transform = transforms.Compose([
                    PairedNoneZeroRegion3D(),
                    PairedRandomCrop3D(cfg["crop_size"]),
                    transforms.RandomChoice([
                        PairedRandomFlip3D(prob=1.0),
                        # PairedRandomRotation90n3d(prob=1.0),
                        PairedRandomRotation3d(prob=1.0),
                        Identity()
                    ]),
                    PairedStdNormalize() if cfg.get("std_norm", False) else Identity(),
                    PairedPad3D(size_divisor=16),
                    PairedToTensor3d(),
                ])
            else:
                self.transform = transforms.Compose([
                    PairedStdNormalize() if cfg.get("std_norm", False) else Identity(),
                    PairedPad3D(size_divisor=16),
                    PairedToTensor3d(),
                ])
        elif mode == "val":
            self.transform = transforms.Compose([
                # PairedNoneZeroRegion3D(),
                PairedStdNormalize() if cfg.get("std_norm", False) else Identity(),
                PairedPad3D(size_divisor=16),
                PairedToTensor3d(),
            ])
        

    def __getitem__(self, item):
        pd_path = self.pd_images[item]
        wip_path = self.wip_images[item]
        # X, Z, Y
        pd_img = nib.load(pd_path).get_fdata().transpose(0,2,1)[:, :, :, np.newaxis].astype(np.float32) # (X, Y, Z, C)
        wip_img = nib.load(wip_path).get_fdata().transpose(0,2,1)[:, :, :, np.newaxis].astype(np.float32)

        # pd max:  1762.0
        # wip max:  1848.0
        # if self.quantile_clip:
        #     pd_img = np.clip(pd_img, a_min=0.0, a_max=np.quantile(pd_img, 0.95))
        #     wip_img = np.clip(wip_img, a_min=0.0, a_max=np.quantile(wip_img, 0.95))
            
        pd_img, wip_img = self.transform((pd_img, wip_img))
        if torch.max(pd_img) > 0:
            pd_img = pd_img / torch.max(pd_img)
        if torch.max(wip_img) > 0:
            wip_img = wip_img / torch.max(wip_img)
            
        # if self.norm_type == "dataset_max":
        #     pd_img = pd_img / 873.0
        #     wip_img = wip_img / 1848.0
        # elif self.norm_type == "sample_max":
        #     if torch.max(pd_img) > 0:
        #         pd_img = pd_img / torch.max(pd_img)
        #     if torch.max(wip_img) > 0:
        #         wip_img = wip_img / torch.max(wip_img)
        # pd_img = torch.tensor(pd_img).permute(3, 0, 1, 2).float() / 1762.0
        # wip_img = torch.tensor(wip_img).permute(3, 0, 1, 2).float() / 1848.0
        if self.return_name:
            return pd_img, wip_img, wip_path
        else:
            return pd_img, wip_img

    def __len__(self):
        return len(self.pd_images)