with open("work_dirs/unet_3d/unet3d_cosinlr_new/20230924_170032.txt", "r") as f:
    lines = f.readlines()

psnr = []
for line in lines:

    if "***** Evaluation *****" in line:
        begin = line.find("SSIM: ") + len("SSIM: ")
        psnr.append(float(line[begin: begin + 4]))


from matplotlib import pyplot as plt
plt.plot(list(range(1, len(psnr) + 1)), psnr)
plt.xlabel("Epoch")
plt.ylabel("ssim")
plt.legend()
plt.savefig("work_dirs/unet_3d/unet3d_cosinlr_new/ssim.png")