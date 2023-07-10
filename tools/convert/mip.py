import numpy as np
import SimpleITK as sitk
import os
from os import listdir
from os.path import isfile, join
import tqdm
import argparse

# %matplotlib inline
parser = argparse.ArgumentParser(description='turn the nifti file into mip')
parser.add_argument('--src-dir',type=str)
parser.add_argument('--out-dir',type=str)
parser.add_argument('--slices_num', type=int, default=16)

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def create_mip(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    #np_img = np_img.transpose(1, 0, 2)
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i - slices_num)
        np_mip[i, :, :] = np.amin(np_img[start:i + 1], 0)
    return np_mip#.transpose(1, 0, 2)


def main():

    for filename in tqdm.tqdm(listdir(args.src_dir)):
        path = join(args.src_dir, filename)
        if filename.endswith('nii.gz'):

            sitk_img = sitk.ReadImage(path)
            np_img = sitk.GetArrayFromImage(sitk_img)
            np_mip = create_mip(np_img, args.slices_num)
            sitk_mip = sitk.GetImageFromArray(np_mip)
            sitk_mip.SetOrigin(sitk_img.GetOrigin())
            sitk_mip.SetSpacing(sitk_img.GetSpacing())
            sitk_mip.SetDirection(sitk_img.GetDirection())
            writer = sitk.ImageFileWriter()
            writer.SetFileName(join(args.out_dir, filename))
            writer.Execute(sitk_mip)


if __name__ == '__main__':
    main()