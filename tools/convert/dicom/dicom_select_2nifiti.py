import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import tqdm
import dicom2nifti
import shutil
import argparse

parser = argparse.ArgumentParser(description='Medical image segmentation in 3D')
parser.add_argument('--area', type=str, required=True)
args = parser.parse_args()

area = args.area
print(f"converting {area}")
for i, case_name in enumerate(tqdm.tqdm(os.listdir(os.path.join('Z:/01_ARIC_V5_Images/ARIC_v5_Images/', area)))):
    nfti_dest = os.path.join('Z:/01_ARIC_V5_Images/nfiti/', f"{area}_{case_name}_pd_space.nii.gz")
    if os.path.exists(nfti_dest):
        continue
    has_pd = False
    desc = set()
    try:
        for file_name in os.listdir(os.path.join('Z:/01_ARIC_V5_Images/ARIC_v5_Images/', area, case_name)):
            # file_path = os.path.join('Z:/01_ARIC_V5_Images/ARIC_v5_Images/', area, case_name, file_name)
            file_path = "Z:/01_ARIC_V5_Images/ARIC_v5_Images/" + area + "/" + case_name + "/" + file_name
            im = imageio.imread(file_path)
            if "SeriesDescription" in im.meta:
                if im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1800ms_linear".lower() or \
                im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1800ms".lower() or \
                    im.meta['SeriesDescription'].lower().strip() == "space668_0.5iso".lower() \
                        or im.meta['SeriesDescription'].lower().strip() == "PD_SPACE_0.5".lower() or \
                            im.meta['SeriesDescription'].lower().strip() == "New_pd_spc_cor_p2_iso_.5mm_770ms".lower() \
                                or im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1300".lower() \
                                    or im.meta['SeriesDescription'].lower().strip() == "3D Cor SPACE".lower():
                        
                    dest = os.path.join("Z:/01_ARIC_V5_Images/pd_images", area, case_name, file_name)
                    os.makedirs(os.path.join("Z:/01_ARIC_V5_Images/pd_images", area, case_name), exist_ok=True)
                    shutil.copyfile(file_path, dest)
                    has_pd = True
                desc.add(im.meta['SeriesDescription'])
        if not has_pd:
            print(f"case {case_name} do not have pd desc!")
            print(desc)
            continue
        if (i + 1) % 20 == 0:
            print(f"converted {i+1} in {area}")
    
        dicom2nifti.dicom_series_to_nifti(os.path.join('Z:/01_ARIC_V5_Images/pd_images/', area, case_name), nfti_dest, reorient_nifti=True)
    except Exception as e:
        print(e)
        print(os.path.join('Z:/01_ARIC_V5_Images/pd_images/', area, case_name), " case fails to convert to nifti")
        continue