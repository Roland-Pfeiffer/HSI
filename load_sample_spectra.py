#!/usr/bin/env python3
import HSI
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

def load_spectra():

    fpath = '/media/findux/DATA/Code/SpectraProcessing_2020-12-09/Reference Spectra/'
    fpath = '/media/findux/DATA/HSI_Data/reference_spectra_josef/'
    files = sorted(os.listdir(fpath))
    fpaths = [fpath + file for file in files]
    material_names = [name.split('.')[0] for name in files]

    print('Shapes:')
    [print(np.loadtxt(file, delimiter=';').shape) for file in fpaths]
    print('Wavelength range limits:')
    for i, file in enumerate(fpaths):
        array = np.loadtxt(file, delimiter=';')
        print(material_names[i], '\t', min(array[:, 0]), max(array[:, 0]))

    wavelengths = np.loadtxt(fpaths[0], delimiter=';')[:, 0]
    # Create numpy array ignoring the second material
    print('Second material ignored)')
    data = np.loadtxt(fpaths[0], delimiter=';')[:, 1]
    for file in fpaths[2:]:
        new_data = np.loadtxt(file, delimiter=';')[:, 1]
        data = np.vstack((data, new_data))
    data = data.T
    return data, wavelengths, material_names