import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spectral

import HSI

# fname = '/media/findux/3C901E14901DD56C/imec/sample data/sample_data_snapshot_vnir_airborne.hdr'
# hdr = spectral.open_image(fname)
# x = np.array(hdr.bands.centers)
# y = np.ones(len(x))
# print(f'Bin count: {len(x)}')
# plt.scatter(x, y)
# plt.show()

print(HSI.wavelen_to_wavenum(1600))
print(HSI.wavenum_to_wavelen(6250))