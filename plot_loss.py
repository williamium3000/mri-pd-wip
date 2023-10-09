with open("work_dirs/new_data/unet3d-b32_cosinlr_50e_sample-max_crop128_clean/20230915_011032.txt", "r") as f:
    lines = f.readlines()

losses = []
val_losses = []
for line in lines:
    if "Epoch" in line and "total loss" in line:
        begin = line.find("total loss: ") + len("total loss: ")
        losses.append(float(line[begin: begin + 5]))
    if "***** Evaluation *****" in line:
        begin = line.find("loss: ") + len("loss: ")
        val_losses.append(float(line[begin: begin + 5]))

print(len(val_losses), len(losses))

from matplotlib import pyplot as plt
plt.plot(list(range(1, len(val_losses) + 1)), losses[::4], label="train loss")
plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("work_dirs/new_data/unet3d-b32_cosinlr_50e_sample-max_crop128_clean/loss.png")