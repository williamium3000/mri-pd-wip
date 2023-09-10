import argparse
import os.path as osp
import nibabel as nib
import numpy as np
import tqdm
import os

class RemoveBackgroundSlices():
    """Remove slices contraining background label. This only works on 3D images (4D tensor)

    """

    def __init__(self, remove_id=0):
        self.remove_id = remove_id

    def __call__(self, input):
        # input X, Y, Z, C
        assert len(input.shape) == 4, "RemoveBackgroundSlices only works on 3D images (4D tensor)"
        
        # drop slices without any label
        z_mask = np.any(input != self.remove_id, axis=(0, 1, 3))

        return input[:, :, z_mask, :]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert 3D medical images to 2D slices.')
    parser.add_argument('pd', type=str, help='image dir')
    parser.add_argument('--out-pd', dest="out_pd", help='output dir for ann')
    parser.add_argument('--out-list', dest="out_list", help='output list')
    args = parser.parse_args()
    return args

def get_files(dir_source):
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
    srcfiles.sort()
    return srcfiles

def main():
    args = parse_args()
    os.makedirs(args.out_pd, exist_ok=True)
    os.makedirs(osp.dirname(args.out_list), exist_ok=True)
    filenames = []
    all_pd_imgs = []
    pd_files = get_files(args.pd)
    for filename_pd in tqdm.tqdm(
        pd_files
    ):
        pd_img = nib.load(osp.join(args.pd, filename_pd)).get_fdata().transpose(0,2,1)
        if len(np.argwhere(np.isinf(pd_img))) > 0:
            for xyz in np.argwhere(np.isinf(pd_img)):
                pd_img[xyz[0], xyz[1], xyz[2]] = 0
        pd_img = pd_img[:, :, :, np.newaxis] # X, Y, Z, C
        
        pd_img = RemoveBackgroundSlices()(pd_img)
        # X, Y, Z, C
        for i in range(pd_img.shape[-2]):
            pd_img_i = pd_img[:, :, i, :]
            np.save(osp.join(args.out_pd, f"{osp.basename(filename_pd)}_{i}.npy"), pd_img_i)
            
            filenames.append(f"{osp.basename(filename_pd)}_{i}.npy\n")
        
        all_pd_imgs.append(pd_img.reshape(-1))
    
    all_pd_imgs = np.concatenate(all_pd_imgs)
    
    pd_mean = np.mean(all_pd_imgs)
    pd_std = np.std(all_pd_imgs)
        
    pd_mim = np.min(all_pd_imgs)
    pd_max = np.max(all_pd_imgs)
    
    
    print("pd stats:")
    print("pd mim: ", pd_mim)
    print("pd max: ", pd_max)
    print("pd mean: ", pd_mean)
    print("pd std: ", pd_std)
    
    print(f"===> writing filenames to {args.out_list}")
    with open(args.out_list, "w") as f:
        f.writelines(filenames)
    
        

    
    
if __name__ == '__main__':
    main()