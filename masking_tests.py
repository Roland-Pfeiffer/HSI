#!/usr/bin/env python3

import spectral

fpath = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

hdr = spectral.open_image(fpath)

with open(fpath) as f:
    lines = f.readlines()
coords = lines[-2:]
print(coords)