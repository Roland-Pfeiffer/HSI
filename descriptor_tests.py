#!/usr/bin/env python3

import HSI
import spectral

fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

# hdr, img, wlv = HSI.load_hsi(fname)
hdr = spectral.open_image(fname)
img = hdr.load()






print('EOF REACHED')