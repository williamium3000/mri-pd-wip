import os
import nibabel as nib
import numpy as np
import tqdm

os.makedirs("work_dirs/meili/result_20_transposed", exist_ok=True)
for file in tqdm.tqdm(os.listdir("work_dirs/meili/result_20")):
    wip_img = nib.load(os.path.join("work_dirs/meili/result_20", file)).get_fdata().transpose(0,2,1)
    nii = nib.Nifti1Image(wip_img, np.eye(4)) 
    nib.save(nii, os.path.join("work_dirs/meili/result_20_transposed", file))