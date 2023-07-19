import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import tqdm
import dicom2nifti
import shutil
import argparse
desc = set()
    
for file_name in tqdm.tqdm(os.listdir('Z:\\04_ARIC_V6_Images\\W-Hagerstown\\W298312\\V6')):
    file_path = os.path.join('Z:\\04_ARIC_V6_Images\\W-Hagerstown\\W298312\\V6', file_name)
    if os.path.isdir(file_path):
        continue
    im = imageio.imread(file_path)
    if "SeriesDescription" in im.meta:
        desc.add(im.meta['SeriesDescription'])
print(desc)