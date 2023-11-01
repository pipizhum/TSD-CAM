import matplotlib.pyplot as plt
import torch
import os
import imageio
import numpy as np
# cividis inferno viridis
color_map = plt.get_cmap("viridis")

coord_shape = [40, 40]  # [2, 40, 40, 2]
coords1 = torch.rand(coord_shape)
# coords1 = coords1.mean(0)
sal_g = color_map(coords1)[:, :, :3] * 255
imageio.imsave(os.path.join("1.jpg"), sal_g.astype(np.uint8))