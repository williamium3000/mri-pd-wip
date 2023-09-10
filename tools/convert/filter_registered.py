import shutil
import os
import tqdm

for file in tqdm.tqdm(os.listdir("data/new_pd_wip/v6/wip_registration")):
    file_size = os.path.getsize(os.path.join("data/new_pd_wip/v6/wip_registration", file)) / 1024 / 1024
    if file_size < 2:
        os.remove(os.path.join("data/new_pd_wip/v6/wip_registration", file))
    