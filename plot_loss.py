with open("work_dirs/unet_3d/unet3d_cosinlr_new/20230924_170032.txt", "r") as f:
    lines = f.readlines()

losses = []
val_losses = []
for line in lines:
    if "Iters: 70/ 75" in line and "total loss" in line:
        begin = line.find("total loss: ") + len("total loss: ")
        losses.append(float(line[begin: begin + 5]))
    if "***** Evaluation *****" in line:
        begin = line.find("total loss: ") + len("total loss: ")
        val_losses.append(float(line[begin: begin + 5]))

print(len(val_losses), len(losses))

from matplotlib import pyplot as plt
plt.plot(list(range(1, len(val_losses) + 1)), losses, label="train loss")
plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("work_dirs/unet_3d/unet3d_cosinlr_new/loss.png")