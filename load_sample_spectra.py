#!/usr/bin/env python3
import HSI
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import pandas as pd
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


def load_samples_in_set(fpath):
    samples = []
    fpaths, files = get_files(fpath)
    for path in fpaths:
        file = pd.read_csv(path, delimiter=',|;', engine='python')  # python engine allows two separators with "or" (|)
        file = np.array(file)
        WLV = file[:, 0]
        intensities = file[:, 1]
        name = os.path.split(path)[1].split('.')[0]
        print('Reading ' + name)
        mat_col = [name for i in range(intensities.shape[0])]
        samples.append(HSI.Spectra(intensities, WLV,mat_col))
    return samples