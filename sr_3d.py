import argparse
import logging
import os
import pprint
import warnings

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml

from model.backbone.unet3d import UNet3D
from util.utils import count_params, init_log
from util.scheduler import *
from util.dist_helper import setup_distributed

from eval3d import evaluate_3d
from dataset.pd_wip_3d import PDWIP3DDataset
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image import StructuralSimilarityIndexMeasure

parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--save_feq', type=int, default=None)
parser.add_argument('--clip-grad-norm', default=None, type=float)
parser.add_argument('--save', action="store_true")
parser.add_argument('--amp', action="store_true")

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    if args.save:
        args.out_dir = os.path.join(args.save_path, "results")
        os.makedirs(args.out_dir, exist_ok=True)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)
    args.rank = rank
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    
    model = UNet3D(**cfg["model"])
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    if cfg["optim"] == "SGD":
        optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg["optim"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    elif cfg["optim"] == "Adam":
        optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4, betas=(0.5, 0.999))
    else:
        raise NotImplementedError(f'{cfg["optim"]} not implemented')
    
    trainset = PDWIP3DDataset(
        cfg=cfg,
        mode="train",
        pd_root=cfg['train_pd_root'],
        wip_root=cfg['train_wip_root'],
        list=args.train_id_path)
    
    valset = PDWIP3DDataset(
        cfg=cfg,
        mode="val",
        pd_root=cfg['val_pd_root'],
        wip_root=cfg['val_wip_root'],
        list=args.val_id_path, return_name=True)
    
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1,
                             pin_memory=True, num_workers=2, drop_last=False, sampler=valsampler)
    
    total_iters = len(trainloader) * cfg['epochs']
    if "scheduler" not in cfg:
        logger.info("no scheduler used")
        scheduler = None
    elif cfg["scheduler"]["name"] == "PolynomialLR":
        logger.info("using PolynomialLR scheduler")
        scheduler = PolynomialLR(optimizer=optimizer, total_iters=total_iters, **cfg["scheduler"]["kwargs"])
    elif cfg["scheduler"]["name"] == "WarmupCosineSchedule":
        logger.info("using WarmupCosineSchedule scheduler")
        scheduler = WarmupCosineSchedule(optimizer=optimizer, t_total=total_iters, **cfg["scheduler"]["kwargs"])
    elif cfg["scheduler"]["name"] == "CosineAnnealingLR":
        logger.info("using CosineAnnealingLR scheduler")
        cfg["scheduler"]["kwargs"]["T_max"] = total_iters
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **cfg["scheduler"]["kwargs"])
    else:
        logger.info(f"using {cfg['scheduler']['name']} scheduler")
        logger.info(cfg["scheduler"])
        scheduler = getattr(lr_scheduler, cfg["scheduler"]["name"])(optimizer=optimizer, **cfg["scheduler"]["kwargs"])


    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    start_epoch = 0
    previous_best = 0.0
    
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    criterionSSIM = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    
    if args.amp:
        scalar = GradScaler()
    else:
        scalar = None
        
    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_ssim = 0.0

        trainsampler.set_epoch(epoch)
        
        for i, (real_A, real_B) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            
            with autocast(enabled=scalar is not None):
                real_A, real_B = real_A.cuda(), real_B.cuda()
                pred_B = model(real_A)
                l1_loss = criterionL1(pred_B, real_B) 
                l2_loss = criterionL2(pred_B, real_B)
                sim_loss = criterionSSIM(pred_B, real_B)
                loss = l1_loss + l2_loss + 10 * sim_loss
            
            total_loss += loss.clone().detach().item()
            total_l1 += l1_loss.clone().detach().item()
            total_l2 += l2_loss.clone().detach().item()
            total_ssim += sim_loss.clone().detach().item()
            #############################################
            #                   update D
            #############################################
            if scalar is not None:
                loss = scalar.scale(loss)
                loss.backward()
                
                if args.clip_grad_norm is not None:
                    scalar.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
                scalar.step(optimizer)
                scalar.update()

            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                optimizer.step()
            

            if "scheduler" in cfg and cfg["lr_decay_per_step"]:
                scheduler.step()

            if ((i % 20) == 0) and (rank == 0):
                logger.info('Iters: {:}/ {:}, lr: {:.6f}, total loss: {:.3f}, l1 loss: {:.3f}, l2 loss: {:.3f}, ssim loss: {:.3f}'.format(
                    i, len(trainloader), optimizer.param_groups[0]['lr'], total_loss / (i+1), 
                    total_l1 / (i+1), 
                    total_l2 / (i+1), 
                    total_ssim / (i+1), 
                    ))

        if "scheduler" in cfg and cfg["lr_decay_per_epoch"]:
            scheduler.step()


        model.eval()
        with torch.no_grad():
            psnr, ssim, mse, l1 = \
                evaluate_3d(args, model, valloader, True)
        
        if rank == 0:
            logger.info(
                '***** Evaluation ***** >>>> AVG(PSNR + SSIM):{:.2f}, PSNR: {:.2f}, SSIM: {:.2f} MSE: {:.4f}, L1: {:.4f}\n'.format(sum([psnr, ssim]) / 2, psnr, ssim, mse, l1))

            if args.save_feq is not None and (epoch + 1) % args.save_feq == 0:
                torch.save({
                    "model": model.module.state_dict()},
                                os.path.join(args.save_path, f'epoch{epoch}.pth'))
            if sum([psnr, ssim]) / 2 > previous_best:
                if os.path.exists(os.path.join(args.save_path, 'best_{:.2f}.pth'.format(previous_best))):
                    os.remove(os.path.join(args.save_path, 'best_{:.2f}.pth'.format(previous_best)))
                previous_best = sum([psnr, ssim]) / 2
                torch.save({
                    "model": model.module.state_dict()},
                                os.path.join(args.save_path, 'best_{:.4f}.pth'.format(previous_best)))
    torch.save({
        "model": model.module.state_dict()},
                    os.path.join(args.save_path, 'last.pth'))


if __name__ == '__main__':
    main()
