import matplotlib.pyplot as plt
import numpy as np
from pyperspectral.load import load_testcube

cube = load_testcube()
rgb: np.ndarray = cube.fake_rgb()

plt.imshow(rgb)
plt.show()