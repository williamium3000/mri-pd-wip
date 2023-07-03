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
        image1, image2 = input[0], input[1]
        assert len(image1.shape) == 4 and len(image2.shape) == 4, "RemoveBackgroundSlices only works on 3D images (4D tensor)"
        
        # drop slices without any label
        z_mask = np.any(image1 != self.remove_id, axis=(0, 1, 3)) & np.any(image2 != self.remove_id, axis=(0, 1, 3))
              
        image1 = image1[:, :, z_mask, :]
        image2 = image2[:, :, z_mask, :]

        return (image1, image2)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert 3D medical images to 2D slices.')
    parser.add_argument('pd', type=str, help='image dir')
    parser.add_argument('wip', type=str, help='annotation dir')
    parser.add_argument('--out-pd', dest="out_pd", help='output dir for ann')
    parser.add_argument('--out-wip', dest="out_wip", help='output dir for img')
    parser.add_argument('--out-list', dest="out_list", help='output list')
    args = parser.parse_args()
    return args

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

def main():
    args = parse_args()
    os.makedirs(args.out_pd, exist_ok=True)
    os.makedirs(args.out_wip, exist_ok=True)
    os.makedirs(osp.dirname(args.out_list), exist_ok=True)
    filenames = []
    all_pd_imgs = []
    all_wip_imgs = []
    pd_files, wip_files = get_files(args.pd, args.wip)
    assert len(wip_files) == len(pd_files)
    for filename_pd, filename_wip in tqdm.tqdm(
        zip(pd_files, wip_files), total=len(wip_files)
    ):
        pd_img = nib.load(osp.join(args.pd, filename_pd)).get_fdata().transpose(0,2,1)
        if len(np.argwhere(np.isinf(pd_img))) > 0:
            for xyz in np.argwhere(np.isinf(pd_img)):
                pd_img[xyz[0], xyz[1], xyz[2]] = 0
        pd_img = pd_img[:, :, :, np.newaxis] # X, Y, Z, C
        
        wip_img = nib.load(osp.join(args.wip, filename_wip)).get_fdata().transpose(0,2,1)
        if len(np.argwhere(np.isinf(wip_img))) > 0:
            for xyz in np.argwhere(np.isinf(wip_img)):
                pd_img[xyz[0], xyz[1], xyz[2]] = 0
        wip_img = wip_img[:, :, :, np.newaxis] # X, Y, Z, C
        
        pd_img, wip_img = RemoveBackgroundSlices()((pd_img, wip_img))
        # X, Y, Z, C
        for i in range(pd_img.shape[-2]):
            pd_img_i = pd_img[:, :, i, :]
            wip_img_i = wip_img[:, :, i, :]
            np.save(osp.join(args.out_pd, f"{osp.basename(filename_pd)}_{i}.npy"), pd_img_i)
            np.save(osp.join(args.out_wip, f"{osp.basename(filename_wip)}_{i}.npy"), wip_img_i)
            
            filenames.append(f"{osp.basename(filename_pd)}_{i}.npy {osp.basename(filename_wip)}_{i}.npy\n")
        
        all_pd_imgs.append(pd_img.reshape(-1))
        all_wip_imgs.append(wip_img.reshape(-1))
    
    all_pd_imgs = np.concatenate(all_pd_imgs)
    all_wip_imgs = np.concatenate(all_wip_imgs)
    
    pd_mean = np.mean(all_pd_imgs)
    pd_std = np.std(all_pd_imgs)
        
    wip_mean = np.mean(all_wip_imgs)
    wip_std = np.std(all_wip_imgs)
    
    pd_mim = np.min(all_pd_imgs)
    pd_max = np.max(all_pd_imgs)
    
    wip_mim = np.min(all_wip_imgs)
    wip_max = np.max(all_wip_imgs)
    
    print("pd stats:")
    print("pd mim: ", pd_mim)
    print("pd max: ", pd_max)
    print("pd mean: ", pd_mean)
    print("pd std: ", pd_std)
    
    print("wip stats:")
    print("wip mim: ", wip_mim)
    print("wip max: ", wip_max)
    print("wip mean: ", wip_mean)
    print("wip std: ", wip_std)
    
    print(f"===> writing filenames to {args.out_list}")
    with open(args.out_list, "w") as f:
        f.writelines(filenames)
    
        

    
    
if __name__ == '__main__':
    main()