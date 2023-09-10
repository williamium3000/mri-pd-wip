with open("work_dirs/unet_2d/unet2d_cosinlr/20230731_094740.txt", "r") as f:
    lines = f.readlines()

losses = []
for line in lines:
    if "total loss" in line:
        begin = line.find("total loss: ") + len("total loss: ")
        losses.append(float(line[begin: begin + 5]))
from matplotlib import pyplot as plt
plt.plot([_ * 5 for _ in range(len(losses))], losses)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("work_dirs/unet_2d/unet2d_cosinlr/loss.png")