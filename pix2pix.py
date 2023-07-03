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

from dataset.pd_wip_2d import PDWIP2DDataset
from model.pix2pix import networks2d
from util.utils import count_params, init_log
from util.scheduler import *
from util.dist_helper import setup_distributed
# from eval3d import evaluate3d

parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--clip-grad-norm', default=None, type=float)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    
    model_G = networks2d.define_G(**cfg["generator"])
    model_D = networks2d.define_D(**cfg["discriminator"])
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model_G) + count_params(model_D)))

    if cfg["optim"] == "SGD":
        optimizer_G = SGD(model_G.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
        optimizer_D = SGD(model_D.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg["optim"] == "AdamW":
        optimizer_G = AdamW(model_G.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
        optimizer_D = AdamW(model_D.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    elif cfg["optim"] == "Adam":
        optimizer_G = Adam(model_G.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        optimizer_D = Adam(model_D.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    else:
        raise NotImplementedError(f'{cfg["optim"]} not implemented')
    
    trainset = PDWIP2DDataset(
        cfg=cfg,
        mode="train",
        pd_root=cfg['pd_root'],
        wip_root=cfg['wip_root'],
        list=args.train_id_path)
    # valset = Medical3DDataset(
    #     cfg=cfg,
    #     mode="val",
    #     img_root=cfg['img_root'],
    #     label_root=cfg['mask_root'],
    #     list=args.val_id_path)
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
    #                        drop_last=False, sampler=valsampler)
    
    total_iters = len(trainloader) * cfg['epochs']
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg["scheduler"]["name"] == "PolynomialLR":
        scheduler_D = PolynomialLR(optimizer=optimizer_D, total_iters=total_iters, **cfg["scheduler"]["kwargs"])
        scheduler_G = PolynomialLR(optimizer=optimizer_G, total_iters=total_iters, **cfg["scheduler"]["kwargs"])
    elif cfg["scheduler"]["name"] == "WarmupCosineSchedule":
        scheduler_D = WarmupCosineSchedule(optimizer=optimizer_D, t_total=total_iters, **cfg["scheduler"]["kwargs"])
        scheduler_G = WarmupCosineSchedule(optimizer=optimizer_G, t_total=total_iters, **cfg["scheduler"]["kwargs"])

    local_rank = int(os.environ["LOCAL_RANK"])
    model_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_G)
    model_D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_D)
    model_G.cuda(local_rank)
    model_D.cuda(local_rank)
    model_G = torch.nn.parallel.DistributedDataParallel(model_G, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    model_D = torch.nn.parallel.DistributedDataParallel(model_D, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    start_epoch = 0
    previous_best = 0.0
    
    criterionGAN = networks2d.GANLoss("lsgan")
    criterionL1 = torch.nn.L1Loss()
    
    
    
    
    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer_G.param_groups[0]['lr'], previous_best))
        total_loss_D = 0.0
        total_loss_G = 0.0

        trainsampler.set_epoch(epoch)
        for i, (real_A, real_B) in enumerate(trainloader):
            model_D.train()
            model_G.train()
            
            real_A, real_B = real_A.cuda(), real_B.cuda()
            #############################################
            #                   update D
            #############################################
            optimizer_D.zero_grad()
            fake_B = model_G(real_A)
            fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model_D(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            
            loss_D.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model_D.parameters(), args.clip_grad_norm)
            optimizer_D.step()

            total_loss_D += loss_D.item()
            #############################################
            #                   update G
            #############################################
            optimizer_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = model_D(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * cfg["lambda_L1"]
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model_G.parameters(), args.clip_grad_norm)
            optimizer_G.step()

            total_loss_G += loss_G.item()
            
            if "scheduler" in cfg and cfg["lr_decay_per_step"]:
                scheduler_D.step()
                scheduler_G.step()

            if ((i % 20) == 0) and (rank == 0):
                logger.info('Iters: {:}/ {:}, loss G: {:.3f}, loss D: {:.3f}'.format(i, len(trainloader), total_loss_G / (i+1), total_loss_D / (i+1)))

        if "scheduler" in cfg and cfg["lr_decay_per_epoch"]:
                scheduler_D.step()
                scheduler_G.step()
        # mIOU, iou_class, fDice, dice_class, fHD95, hd95_class = \
        #     evaluate3d(model, valloader, cfg, local_rank)

        # if rank == 0:
        #     logger.info(
        #         '***** Evaluation {} ***** >>>> mIOU: {:.2f}, fDice: {:.2f} fHD95: {:.2f}\n'.format(
        #             cfg["eval_mode"], mIOU, fDice, fHD95))

        # if fDice > previous_best and rank == 0:
        #     if previous_best != 0:
        #         os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone']["name"], previous_best)))
        #     previous_best = fDice
        #     torch.save(model.module.state_dict(),
        #                os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone']["name"], fDice)))
    torch.save({
        "model_G": model_G.module.state_dict(),
        "model_D": model_D.module.state_dict()},
                    os.path.join(args.save_path, 'last.pth'))


if __name__ == '__main__':
    main()
