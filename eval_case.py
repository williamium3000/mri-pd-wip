import argparse
import logging
import os
import pprint
import warnings

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import SGD, AdamW, Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml
import numpy as np
import nibabel as nib
import tqdm

from dataset.pd_wip_3d import PDWIP3DDataset
from model.pix2pix import networks2d
from util.utils import count_params, init_log
from util.scheduler import *
from util.dist_helper import setup_distributed
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError


parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--case_dir', type=str, required=True)
parser.add_argument('--pd_root', type=str, required=True)
parser.add_argument('--wip_root', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)


def evaluate_2d(args, dataloader):
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    mean_squared_error = MeanSquaredError()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_num = 0
    eval_files = list(os.listdir(args.case_dir))
    
    for i, (real_A, real_B, real_B_name) in enumerate(tqdm.tqdm(dataloader)):

        case_name = real_B_name[0]
        case_id = case_name.split("-")[0]
        for eval_file in eval_files:
            if case_id in eval_file:
                break
        output = nib.load(os.path.join(args.case_dir, eval_file)).get_fdata().transpose(0,2,1)[:, :, :, np.newaxis] # (X, Y, Z, C)
        output = torch.tensor(output).permute(3, 0, 1, 2).float()
        output = output.unsqueeze(0)
        total_psnr += psnr(real_B, output)
        total_ssim += ssim(real_B, output)
        total_mse += mean_squared_error(real_B, output)
        total_l1 += (torch.abs(real_B - output)).mean()
        total_num += 1
        
    psnr, ssim, mse, l1 = total_psnr / total_num, total_ssim / total_num, total_mse / total_num, total_l1 / total_num

    return psnr, ssim, mse, l1

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    logger.info('{}\n'.format(pprint.pformat(cfg)))

    cudnn.enabled = True
    cudnn.benchmark = True

    
    valset = PDWIP3DDataset(
        cfg=cfg,
        mode="val",
        pd_root=args.pd_root,
        wip_root=args.wip_root,
        list=args.val_id_path, return_name=True)
    
    valloader = DataLoader(valset, batch_size=1,
                             pin_memory=True, num_workers=2, drop_last=True, shuffle=False)
    psnr, ssim, mse, l1 = evaluate_2d(args, valloader)
    print('***** Evaluation ***** >>>> PSNR: {:.4f}, SSIM: {:.4f} MSE: {:.4f}, L1: {:.4f}\n'.format(psnr, ssim, mse, l1))
    
        
        
        

if __name__ == '__main__':
    main()
