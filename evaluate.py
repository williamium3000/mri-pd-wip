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
from model.builder import build_model
from util.utils import count_params, init_log
from util.scheduler import *
from util.dist_helper import setup_distributed
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError


parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--pd_root', type=str, required=True)
parser.add_argument('--wip_root', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--g-key', type=str, default="model", required=True)
parser.add_argument('--save', action="store_true")
parser.add_argument('--eval-3d', action="store_true")


def evaluate_3d(args, model_G, dataloader, dist_eval):
    psnr = PeakSignalNoiseRatio().cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    mean_squared_error = MeanSquaredError().cuda()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_num = 0
    
    for i, (real_A, real_B, real_B_name) in enumerate(tqdm.tqdm(dataloader)):
        real_A, real_B = real_A.cuda(), real_B.cuda() # 1, 1, 320, 320, 128
        fake_B = model_G(real_A)
        total_psnr += psnr(real_B, fake_B)
        total_ssim += ssim(real_B, fake_B)
        total_mse += mean_squared_error(real_B, fake_B)
        total_l1 += (torch.abs(real_B - fake_B)).mean()
        total_num += 1
        
        if args.save:
            fake_B = fake_B.squeeze(0).squeeze(0).detach().cpu().permute(0, 2, 1).numpy()
            ori_data = nib.load(real_B_name[0]) 
            nii = nib.Nifti1Image(fake_B, ori_data.affine, ori_data.header) 
            nib.save(nii, os.path.join(args.out_dir, os.path.basename(real_B_name[0])))
    
    if dist_eval:
        dist.all_reduce(total_psnr)
        dist.all_reduce(total_ssim)
        dist.all_reduce(total_mse)
        dist.all_reduce(total_l1)
        total_num = torch.tensor(total_num).cuda()
        dist.all_reduce(total_num)
        dist.barrier()
    
    psnr, ssim, mse, l1 = total_psnr / total_num, total_ssim / total_num, total_mse / total_num, total_l1 / total_num
    
    return psnr, ssim, mse, l1

def evaluate_2d(args, model, dataloader, dist_eval):
    psnr = PeakSignalNoiseRatio().cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    mean_squared_error = MeanSquaredError().cuda()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_num = 0
    
    for i, (real_A, real_B, real_B_name) in enumerate(tqdm.tqdm(dataloader)):

        real_A, real_B = real_A.cuda(), real_B.cuda() # 1, 1, 320, 320, 128
        real_A = real_A.squeeze(0).permute(3, 0, 1, 2) # 128, 1, 320, 320

        fake_B = model(real_A) # 128, 1, 320, 320
        
        output = fake_B.permute(1, 2, 3, 0) # 1, 320, 320, 128
        output = output.unsqueeze(0) # 1, 1, 320, 320, 128
        total_psnr += psnr(real_B, output)
        total_ssim += ssim(real_B, output)
        total_mse += mean_squared_error(real_B, output)
        total_l1 += (torch.abs(real_B - output)).mean()
        total_num += 1
        
        if args.save:
            fake_B = fake_B.permute(1, 2, 3, 0).detach().cpu().squeeze(0).permute(0, 2, 1).numpy()
            ori_data = nib.load(real_B_name[0]) 
            nii = nib.Nifti1Image(fake_B, ori_data.affine, ori_data.header) 
            nib.save(nii, os.path.join(args.out_dir, os.path.basename(real_B_name[0])))
    
    
    if dist_eval:
        dist.all_reduce(total_psnr)
        dist.all_reduce(total_ssim)
        dist.all_reduce(total_mse)
        dist.all_reduce(total_l1)
        total_num = torch.tensor(total_num).cuda()
        dist.all_reduce(total_num)
        dist.barrier()
        
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

    
    model, _ = build_model(cfg)
    
    valset = PDWIP3DDataset(
        cfg=cfg,
        mode="val",
        pd_root=args.pd_root,
        wip_root=args.wip_root,
        list=args.val_id_path, return_name=True)
    
    valloader = DataLoader(valset, batch_size=1,
                             pin_memory=True, num_workers=2, drop_last=True, shuffle=False)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")[args.g_key])
    
    model.cuda()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    
    with torch.no_grad():
        psnr, ssim, mse, l1 = evaluate_3d(args, model, valloader, False) if args.eval_3d else evaluate_2d(args, model, valloader, False)
    print('***** Evaluation ***** >>>> PSNR: {:.4f}, SSIM: {:.4f} MSE: {:.4f}, L1: {:.4f}\n'.format(psnr, ssim, mse, l1))
    
        
        
        

if __name__ == '__main__':
    main()
