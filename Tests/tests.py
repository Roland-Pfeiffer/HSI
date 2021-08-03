import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spectral

import HSI
import load_sample_spectra

fname = '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_007/capture/2019-07-16_007.hdr'
fname_masl = '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_007/2019-07-16_007_BG.tif'
hdr = spectral.open_image(fname)
x = np.array(hdr.bands.centers)
print(x[0], x[-1])

# y = np.ones(len(x))
# print(f'Bin count: {len(x)}')
# plt.scatter(x, y)
# plt.show()

# start_WN = 400
# stop_WN = 2300
# start_WL = HSI.wavenum_to_wavelen(stop_WN)  # Wavenumbers are inverted to wavelength
# stop_WL = HSI.wavenum_to_wavelen(start_WN)  # Wavenumbers are inverted to wavelength
# print(f'Start: {start_WL:.2f} nm | Stop: {stop_WL:.2f} nm')
#
# a = [1, 2, 3, 4]
# print([a])

fpath = '/media/findux/DATA/spectral_data/reference_spectra_josef/'
load_sample_spectra.show_csv_structure(fpath)