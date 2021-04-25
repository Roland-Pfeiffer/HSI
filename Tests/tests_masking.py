#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import spectral
import HSI



np.set_printoptions(edgeitems=100, linewidth=220)

fpath_hsi = ('/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/'
             '2019-07-16_003/capture/2019-07-16_003.hdr')
hdr = spectral.open_image(fpath_hsi)
path = '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_003/'
file = path.split('/')[-2]
mask_path = path + file + '_LITTER.tif'
mask = HSI.BinaryMask(mask_path, 'Litter')

hdr, cube, wlv = HSI.load_hsi(fpath_hsi)
data_unf = HSI.unfold_cube(cube)
data_plastic = data_unf[mask.in_value_indices, :]

data_plastic = HSI.Spectra(data_plastic, wlv)
plt.plot(wlv, data_plastic.T)
plt.show()

