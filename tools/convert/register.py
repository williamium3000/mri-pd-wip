import os
from tqdm import tqdm

tool = "~/fsl/bin/flirt"
params = "-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear"

pd_dir = "data/new_pd_wip/v6/pd_paired"
wip_dir = "data/new_pd_wip/v6/wip_space"
out_dir = "data/new_pd_wip/v6/wip_registration"
omat_dir = "data/new_pd_wip/v6/omat_wip_registration"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(omat_dir, exist_ok=True)

pd_files = list(os.listdir(pd_dir))
wip_files = list(os.listdir(wip_dir))

pd_case_file_mapping = {filename.split("_")[0]:filename for filename in pd_files}
wip_case_file_mapping = {filename.split("_")[0]:filename for filename in wip_files}

pd_case_names = set([filename.split("_")[0] for filename in pd_files])
wip_case_names = set([filename.split("_")[0] for filename in wip_files])

paired = list(pd_case_names.intersection(wip_case_names))

for case_name in tqdm(paired):
    pd_file_path = os.path.join(pd_dir, pd_case_file_mapping[case_name])
    wip_file_path = os.path.join(wip_dir, wip_case_file_mapping[case_name])
    outname = wip_case_file_mapping[case_name].split(".")[0]+"_registration"
    if os.path.exists(os.path.join(out_dir, outname + ".nii.gz")):
        print("skipping ", os.path.join(out_dir, outname + ".nii.gz"))
        continue
    
    out = os.path.join(out_dir, outname)
    omatname = outname+".mat"
    omat = os.path.join(omat_dir, omatname)
    
    cmd = tool + " -in " + wip_file_path +" -ref " + pd_file_path + " -out " + out + " -omat " + omat + " " + params
    ret = os.popen(cmd).read()
    print(ret)