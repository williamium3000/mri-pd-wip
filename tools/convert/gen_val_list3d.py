import argparse
import os.path as osp
import nibabel as nib
import numpy as np
import tqdm
import os

parser = argparse.ArgumentParser(
        description='Convert 3D medical images to 2D slices.')
parser.add_argument('--pd', type=str, help='image dir')
parser.add_argument('--wip', type=str, help='annotation dir')
parser.add_argument('--out-list', dest="out_list", help='output list')
args = parser.parse_args()

def get_files(dir_source, dir_target):
    """
    # helper function that returns tiff files to process
    inputs:
    dirsource  folder containing input tiff files
    dirtarget  folder containing output tiff files

    outputs:   srcfiles, tgtfiles
    """
    srcfiles = []
    for fname in os.listdir(dir_source):
        if fname.endswith('.gz') > 0 or fname.endswith('.tiff'):
            srcfiles.append(fname)
    tgtfiles = []
    for fname in os.listdir(dir_target):
        if fname.endswith('.gz') > 0 or fname.endswith('.tiff'):
            tgtfiles.append(fname)
    srcfiles.sort()
    tgtfiles.sort()
    return srcfiles, tgtfiles
lines = []
pd_files, wip_files = get_files(args.pd, args.wip)
for filename_pd, filename_wip in tqdm.tqdm(
        zip(pd_files, wip_files), total=len(wip_files)
    ):
    lines.append(f"{filename_pd} {filename_wip}\n")

with open(args.out_list, 'w') as f:
    f.writelines(lines)