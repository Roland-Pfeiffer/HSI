#!/usr/bin/env python3
import HSI
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

def load_spectra(plot=False, verbose=False):

    fpath = '/media/findux/DATA/HSI_Data/reference_spectra_josef/'
    files = sorted(os.listdir(fpath))
    fpaths = [fpath + file for file in files]
    material_names = [name.split('.')[0] for name in files]

    if verbose:
        print('Shapes:')
        [print(np.loadtxt(file, delimiter=';').shape) for file in fpaths]
        print('Wavelength range limits:')
        for i, file in enumerate(fpaths):
            array = np.loadtxt(file, delimiter=';')
            array = array.round(4)
            print(material_names[i].ljust(6), min(array[:, 0]), max(array[:, 0]), '\tLength:', array.shape[0])

        print('Dropping PE.')
    PE_i = material_names.index('PE')
    del material_names[PE_i]
    del fpaths[PE_i]

    wavelengths = np.loadtxt(fpaths[0], delimiter=';')[:, 0]
    # Create numpy array ignoring the second material
    data = np.loadtxt(fpaths[0], delimiter=';')[:, 1]
    logging.debug(data)
    for file in fpaths[1:]:
        new_data = np.loadtxt(file, delimiter=';')[:, 1]
        data = np.vstack((data, new_data))
        logging.debug(data)

    data_dict = dict(zip(material_names, data.T))
    if verbose:
        print(data_dict)
    if plot:
        plt.figure('Sample spectra')
        for i in range(data.shape[0]):
            plt.plot(wavelengths, data[i].T, label=material_names[i])
            plt.legend()
        plt.show()
    return data, wavelengths, material_names