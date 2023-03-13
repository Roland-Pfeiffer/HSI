#!/usr/bin/env python3
import pyperspectral
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
fpath = '/media/findux/DATA/Code/SpectraProcessing_2020-12-09/Reference Spectra/'
files = sorted(os.listdir(fpath))
fpaths = [fpath + file for file in files]
names = [name.split('.')[0] for name in files]

print('Shapes:')
[print(np.loadtxt(file, delimiter=';').shape) for file in fpaths]
print('Wavelength range limits:')
for i, file in enumerate(fpaths):
    array = np.loadtxt(file, delimiter=';')
    print(names[i], '\t', min(array[:, 0]), max(array[:, 0]))

wavelengths = np.loadtxt(fpaths[0], delimiter=';')[:, 1]
# Create numpy array ignoring the second material
data = np.loadtxt(fpaths[0], delimiter=';')[:, 1]
for file in fpaths[2:]:
    data = np.vstack((data, np.loadtxt(file, delimiter=';')[:, 1]))
data = data.T




plt.plot(wavelengths, data)
plt.show()