#!/usr/bin/env python3

import spectral
import numpy as np

fname = "/media/findux/DATA/HSI_Data/SPECIM_field_data/2019-07-16_HS_images_laid_out" +\
        "/2019-07-16_005/results/REFLECTANCE_2019-07-16_005.hdr"

hdr = spectral.open_image(fname)
print(hdr.shape)

# Wrong. Horizontal layers like RGB layers are not
tc1 = np.array([[[1, 2, 3],
                 [4, 5, 6]],

                [[1, 2, 3],
                 [4, 5, 6]],

                [[1, 2, 3],
                 [4, 5, 6]],

                [[1, 2, 3],
                 [4, 5, 6]]])
print(tc1.shape)

tc2 = np.array([[[1, 2, 3],
                 [1, 2, 3],
                 [1, 2, 3],
                 [1, 2, 3]],

                [[4, 5, 6],
                 [4, 5, 6],
                 [4, 5, 6],
                 [4, 5, 6]]])
print(tc2.shape)

tc3 = np.array([[[1, 1, 1, 1],
                 [2, 2, 2, 2],
                 [3, 3, 3, 3]],
                [[4, 4, 4, 4],
                 [5, 5, 5, 5],
                 [6, 6, 6, 6]]])
dims = tc3.shape
tc3_unf = tc3.reshape((dims[0] * dims[1], dims[2]))
print(tc3_unf)