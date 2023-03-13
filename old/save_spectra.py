#!/usr/bin/env python3
import numpy as np
import pyperspectral
import time


fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'
fname_out_npz = '/home/findux/Desktop/spectra_out.npz'

print("Loading data...")
_, cube, wlv = pyperspectral.load_hsi(fname)

print("Unfolding data...")
spectra = pyperspectral.unfold_cube(cube)

print("Saving NPZ...")
np.savez(fname_out_npy, spectra=spectra, wlv=wlv)
print('NPZ saved.')