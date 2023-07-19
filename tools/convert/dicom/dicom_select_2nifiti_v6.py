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
start_letter = area[0]
print(f"converting {area}")
for i, case_name in enumerate(tqdm.tqdm(os.listdir(os.path.join('Z:\\04_ARIC_V6_Images', area)))):
    if not case_name.startswith(start_letter):
        continue
    
    nfti_dest_pd = os.path.join('Z:\\04_ARIC_V6_Images\\nfiti\\pd', f"{area}_{case_name}_pd_space.nii.gz")
    nfti_dest_wip = os.path.join('Z:\\04_ARIC_V6_Images\\nfiti\\wip', f"{area}_{case_name}_wip_space.nii.gz")
    if os.path.exists(nfti_dest_pd) and os.path.exists(nfti_dest_wip):
        continue
    
    has_pd = False
    has_wip = False
    desc = set()
    
    for file_name in tqdm.tqdm(os.listdir(os.path.join('Z:\\04_ARIC_V6_Images', area, case_name, "V6"))):
        file_path = os.path.join('Z:\\04_ARIC_V6_Images', area, case_name, "V6", file_name)
        im = imageio.imread(file_path)
        if "SeriesDescription" in im.meta:
            if im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1800ms_linear".lower() or \
                im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1800ms".lower() or \
                    im.meta['SeriesDescription'].lower().strip() == "space668_0.5iso".lower() \
                        or im.meta['SeriesDescription'].lower().strip() == "PD_SPACE_0.5".lower() or \
                            im.meta['SeriesDescription'].lower().strip() == "New_pd_spc_cor_p2_iso_.5mm_770ms".lower() \
                                or im.meta['SeriesDescription'].lower().strip() == "pd_spc_cor_p2_iso_.5mm_1300".lower() \
                                    or im.meta['SeriesDescription'].lower().strip() == "3D Cor SPACE".lower():          
                dest = os.path.join("Z:\\04_ARIC_V6_Images\\pd_images", area, case_name, file_name)
                os.makedirs(os.path.join("Z:\\04_ARIC_V6_Images\\pd_images", area, case_name), exist_ok=True)
                shutil.copyfile(file_path, dest)
                has_pd = True
            
            if im.meta['SeriesDescription'].lower().strip() == 'Research WIP_SPACE_0.5'.lower():
                dest = os.path.join("Z:\\04_ARIC_V6_Images\\wip_images", area, case_name, file_name)
                os.makedirs(os.path.join("Z:\\04_ARIC_V6_Images\\wip_images", area, case_name), exist_ok=True)
                shutil.copyfile(file_path, dest)
                has_wip = True
            desc.add(im.meta['SeriesDescription'])
    if not has_pd or not has_wip:
        print(f"case {case_name} do not have pd or wip desc!")
        print(desc)
        continue
    if (i + 1) % 20 == 0:
        print(f"converted {i+1} in {area}")
    
    
    try:
        dicom2nifti.dicom_series_to_nifti(os.path.join('Z:\\04_ARIC_V6_Images\\pd_images\\', area, case_name), nfti_dest_pd, reorient_nifti=True)
    except Exception as e:
        print(e)
        print(os.path.join('Z:\\04_ARIC_V6_Images\\pd_images\\', area, case_name), " case fails to convert to nifti")
    try:
        dicom2nifti.dicom_series_to_nifti(os.path.join('Z:\\04_ARIC_V6_Images\\wip_images\\', area, case_name), nfti_dest_wip, reorient_nifti=True)
    except Exception as e:
        print(e)
        print(os.path.join('Z:\\04_ARIC_V6_Images\\wip_images\\', area, case_name), " case fails to convert to nifti")