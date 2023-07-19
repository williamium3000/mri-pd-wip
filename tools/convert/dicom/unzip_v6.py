from zipfile import ZipFile
from pathlib import Path
import os
import tqdm

dest_root = "Z:\\04_ARIC_V6_Images\\unzip_files"
path = Path("Z:\\04_ARIC_V6_Images\\ZipFiles")
total_files = list(path.rglob("*.zip"))
for p in tqdm.tqdm(total_files):
   case_name = os.path.basename(p).split("-")[0]
   dest_dir = os.path.join(dest_root, case_name)
   if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
      continue
   os.makedirs(dest_dir, exist_ok=True)
   try:
      with ZipFile(p, 'r') as zipObject:
         listOfFileNames = zipObject.namelist()
         wip_list = [fileName for fileName in listOfFileNames if "WIP_SPACE" in fileName]
         pd_list = [fileName for fileName in listOfFileNames if "PD_SPACE" in fileName]
         zipObject.extractall(dest_dir, wip_list + pd_list)
   except Exception as e:
      print(e)
      print(p, "has a problem !")