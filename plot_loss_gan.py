with open("work_dirs/pix2pix_3d/unet3d-b32-residual_patchgan3d-ndf32_cosinlr_crop128_sample-max_50e/20230903_202946.txt", "r") as f:
    lines = f.readlines()

losses_D = []
losses_G = []
for line in lines:
    if "loss" in line:
        begin = line.find("loss D: ") + len("loss D: ")
        losses_D.append(float(line[begin: begin + 5]))
        
        begin = line.find("loss G: ") + len("loss G: ")
        losses_G.append(float(line[begin: begin + 5]))
from matplotlib import pyplot as plt
plt.plot([_ * 5 for _ in range(len(losses_D))], losses_D, label="loss D")
plt.plot([_ * 5 for _ in range(len(losses_G))], losses_G, label="loss G")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.savefig("work_dirs/pix2pix_3d/unet3d-b32-residual_patchgan3d-ndf32_cosinlr_crop128_sample-max_50e/loss.png")