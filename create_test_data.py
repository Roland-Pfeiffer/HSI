#!/usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import spectral
from sklearn.preprocessing import minmax_scale

import HSI

fpaths = ['/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_003/',
          '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_005/',
          '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_007/',
          '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_011/',
          '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_012/']
# for path in fpaths:
# file = path.split('/')[-2]
# mask_path = path + file + '_LITTER.tif'
