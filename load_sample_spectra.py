#!/usr/bin/env python3
import HSI
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import logging

def get_files(fpath):
    files = sorted(os.listdir(fpath))
    if 'BACKUPS' in files:
        files.remove('BACKUPS')
    fpaths = [fpath + file for file in files]
    return fpaths, files

def show_struct(fpath):
    fpaths, files = get_files(fpath)
    for path in fpaths:
        with open(path) as f:
            pst = os.path.split(path)
            print(f.readlines()[2].strip(), pst[-1])

def cleanup(fpath):
    """Turns comma-seperated into """
    fpaths, files = get_files(fpath)
    # Create backups
    if not os.path.isdir(fpath + '/BACKUPS'):
        print('Creating backups.')
        os.mkdir(fpath + '/BACKUPS')
        for path in fpaths:
            shutil.copy(path, os.path.split(path)[0] + '/BACKUPS/' + os.path.split(path)[1])
    else:
        print('Backups already found.')
    # Correct data
    for i in range(len(fpaths)):
        with open(fpaths[i], 'r') as read:
            lines = read.readlines()
        # Comma-separated regex: decimal . decimals e+ 2decimals, possible minus (and then the same again)
        pattern = re.compile(r'[-]?\d\.\d+[e][+-]\d\d[,][-]?\d\.\d+[e][+-]\d\d')
        matched = pattern.search(lines[0])
        print(matched)
        if not matched == None:
            lines = [line.replace(',', ';') for line in lines]
        lines = [line.replace(',', '.') for line in lines]
        with open(fpaths[i], 'w') as write:
            write.writelines(lines)


def load_spectra(fpath: str(), plot=False):
    fpaths, files = get_files(fpath)
    material_names = [name.split('.')[0] for name in files]
    wavelengths = np.loadtxt(fpaths[0], delimiter=';')[:, 0]
    # Create numpy array ignoring the second material
    data = np.loadtxt(fpaths[0], delimiter=';')[:, 1]
    logging.debug(data)
    for file in fpaths[1:]:
        new_data = np.loadtxt(file, delimiter=';')[:, 1]
        data = np.vstack((data, new_data))
        logging.debug(data)
    if plot:
        plt.figure('Sample spectra')
        for i in range(data.shape[0]):
            plt.plot(wavelengths, data[i].T, label=material_names[i])
            plt.legend()
        plt.show()
    return data, wavelengths, material_names