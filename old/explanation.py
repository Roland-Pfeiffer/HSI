#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

fpath = "/media/findux/DATA/Documents/Malta_II/datasets/ir_rgb_overlay/DJI_0515/DJI_0515.JPG"
img = Image.open(fpath)

plt.figure("Fig. 1")

plt.imshow(img)

plt.figure("Fig. 2")
img_array = np.array(img)
print(img_array.shape)

plt.imshow(img_array[:, :, 2], cmap="gray")
plt.show()