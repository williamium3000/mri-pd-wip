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
# from torch.utils.data import DataLoader
import yaml

from dataset.medical3d import Medical3DDataset
from model.semseg.segmentor import Segmentor
from util.loss import ProbOhemCrossEntropy2d, CrossEntropyAndDice
from segmentation_models_pytorch.losses import DiceLoss
from util.utils import count_params, init_log
from util.scheduler import *
from util.dist_helper import setup_distributed
from eval3d import evaluate3d

import hfai
from ffrecord.torch import DataLoader

parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--clip-grad-norm', default=None, type=float)
parser.add_argument('--amp', action="store_true")


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

    model = Segmentor(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    param_group = [{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': model.head.parameters(),
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    if cfg["optim"] == "SGD":
        optimizer = SGD(param_group, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg["optim"] == "AdamW":
        optimizer = AdamW(param_group, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    elif cfg["optim"] == "Adam":
        optimizer = Adam(param_group, lr=cfg['lr'], weight_decay=1e-4)
    else:
        raise NotImplementedError(f'{cfg["optim"]} not implemented')
    
    total_iters = len(trainloader) * cfg['epochs']
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg["scheduler"]["name"] == "PolynomialLR":
        scheduler = PolynomialLR(optimizer=optimizer, total_iters=total_iters, **cfg["scheduler"]["kwargs"])
    elif cfg["scheduler"]["name"] == "WarmupCosineSchedule":
        scheduler = WarmupCosineSchedule(optimizer=optimizer, t_total=total_iters, **cfg["scheduler"]["kwargs"])
    
    try: # old torch does not have the amp
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
    except:
        scaler = None
        args.amp = False
        warnings.warn("current torch does not support fp16!")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    start_epoch = 0
    start_step = 0
    start_epoch, start_step, previous_best = hfai.checkpoint.init(model, optimizer, scheduler=scheduler, amp_scaler=scaler,
                                                             ckpt_path=os.path.join(args.save_path, 'last.pth'))
    previous_best = previous_best if previous_best else 0.0
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'Dice':
        criterion = DiceLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'CrossEntropyAndDice':
        criterion = CrossEntropyAndDice(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = Medical3DDataset(
        cfg=cfg,
        mode="train",
        img_root=cfg['img_root'],
        label_root=cfg['mask_root'],
        list=args.train_id_path)
    valset = Medical3DDataset(
        cfg=cfg,
        mode="val",
        img_root=cfg['img_root'],
        label_root=cfg['mask_root'],
        list=args.val_id_path)
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)
    
    
    
    
    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = 0.0

        trainsampler.set_epoch(epoch)
        trainloader.set_step(start_step)
        
        for i, (img, mask) in enumerate(trainloader):
            i += start_step
            img, mask = img.cuda(), mask.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(img)
                loss = criterion(pred, mask)
            
            torch.distributed.barrier()
            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            
            
            total_loss += loss.item()
            if cfg["lr_decay_per_step"]:
                scheduler.step()

            if ((i % 10) == 0) and (rank == 0):
                logger.info('Iters: {:}/ {:}, Total loss: {:.3f}'.format(i, len(trainloader), total_loss / (i+1)))

            model.try_save(epoch, i + 1, others=previous_best)
        # reset start step
        start_step = 0  
        if cfg["lr_decay_per_epoch"]:
                scheduler.step()
        mIOU, iou_class, fDice, dice_class, fHD95, hd95_class = \
            evaluate3d(model, valloader, cfg, local_rank)

        if rank == 0:
            logger.info(
                '***** Evaluation {} ***** >>>> mIOU: {:.2f}, fDice: {:.2f} fHD95: {:.2f}\n'.format(
                    cfg["eval_mode"], mIOU, fDice, fHD95))

        if fDice > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone']["name"], previous_best)))
            previous_best = fDice
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone']["name"], fDice)))


if __name__ == '__main__':
    main()