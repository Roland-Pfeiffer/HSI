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

start_WN = 400
stop_WN = 2300
start_WL = HSI.wavenum_to_wavelen(stop_WN)  # Wavenumbers are inverted to wavelength
stop_WL = HSI.wavenum_to_wavelen(start_WN)  # Wavenumbers are inverted to wavelength
print(f'Start: {start_WL:.2f} nm | Stop: {stop_WL:.2f} nm')

a = [1, 2, 3, 4]
print([a])