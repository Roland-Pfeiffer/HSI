#!/usr/bin/env python3
"""NOTE: This uses a custom, edited version of the .hdr file. It normally does not contain coordinates.
This was just a test to see if anything breaks when the file is changed and its length changes."""

import spectral

fpath = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

hdr = spectral.open_image(fpath)

with open(fpath) as f:
    lines = f.readlines()
coords = lines[-2:]
print(coords)