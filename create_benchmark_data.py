#!/usr/bin/env python3
"""THIS IS JOSEF'S STUFF"""
import numpy as np
import descriptors

fname = "/home/findux/Desktop/spectra_out.npy"

print("Reading data.")
spectra = np.load(fname)

print("Getting descriptors")
descs = [descriptors.getDescriptorSetForSpec(str(i), spectra[i]) for i in spectra]

print("Done")