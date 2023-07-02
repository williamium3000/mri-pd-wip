import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
import torch.distributed as dist
from dataset.medical3d import Medical3DDataset
from model.semseg.segmentor import Segmentor
from util.dist_helper import setup_distributed
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, Dice, HD95
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def evaluate3d(model, loader, cfg, local_rank=-1):
    mode = cfg["eval_mode"]
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    dice_meter = AverageMeter()
    hd95_meter = AverageMeter()

    with torch.no_grad():
        for img, mask in tqdm.tqdm(loader, disable=local_rank != 0):
            img = img.cuda()
            bs = img.size(0)
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                stride_ratio = cfg['sw_stride_ratio']
                b, _, h, w, z = img.shape
                final = torch.zeros(b, cfg["nclass"], h, w, z).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        dep = 0
                        while dep < z:
                            pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid), dep: min(z, dep + grid)])
                            final[:, :, row: min(h, row + grid), col: min(w, col + grid), dep: min(z, dep + grid)] += pred.softmax(dim=1)
                            dep += int(grid * stride_ratio)
                        col += int(grid * stride_ratio)
                    row += int(grid * stride_ratio)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w, z = img.shape[2:]
                    start_h, start_w, start_z = \
                        (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2, (z - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size'], start_z:start_z + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size'], start_z:start_z + cfg['crop_size']]

                pred = model(img).argmax(dim=1)
            
            # eval IoU
            pred, mask = pred.cpu().numpy(), mask.numpy()
            intersection, union, _ = \
                intersectionAndUnion(pred.reshape(bs, -1), mask.reshape(bs, -1), cfg['nclass'], 255)
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            
            # eval Dice & HD95
            dice = Dice(pred, mask, cfg['nclass']) # array of (nclass, )
            hd95 = HD95(pred, mask, cfg['nclass']) # array of (nclass, )
            reduced_dice = torch.from_numpy(dice).cuda()
            reduced_hd95 = torch.from_numpy(hd95).cuda()
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dice_meter.update(reduced_dice.cpu().numpy())
            hd95_meter.update(reduced_hd95.cpu().numpy())

    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0

    dice_class = dice_meter.avg
    fDice = np.mean(dice_class[1:]) * 100.0
    
    hd95_class = hd95_meter.avg
    fHD95 = np.mean(hd95_class[1:])
    return mIOU, iou_class, fDice, dice_class, fHD95, hd95_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        print('{}\n'.format(pprint.pformat(cfg)))


    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    # print(model)
    if rank == 0:
        print('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    if cfg['dataset'] == 'cityscapes':
        eval_mode = 'sliding_window'
    else:
        eval_mode = 'original'
    mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg, local_rank)
    if rank == 0:
        print('***** Evaluation {} ***** >>>> meanIOU: {:.3f}\n'.format(eval_mode, mIOU))
        iou_class = [(cls_idx, iou) for cls_idx, iou in enumerate(iou_class)]
        iou_class.sort(key=lambda x:x[1])
        for (cls_idx, iou) in iou_class:
            print('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou * 100, iou_class[cls_idx] * 100))

if __name__ == '__main__':
    main()
