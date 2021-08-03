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
    """Returns a list with file paths of every file in the folder, and a list with all file names."""
    files = sorted(os.listdir(fpath))
    if 'BACKUPS' in files:
        files.remove('BACKUPS')
    fpaths = [fpath + file for file in files]
    return fpaths, files


def show_csv_structure(fpath):
    """Shows the structure of the files within a given directory by printing the third line of each file.
    Assumes that the files are csv files.
    Also assumes that there are no other directories. Ignores a BACKUPS directory if present."""
    fpaths, files = get_files(fpath)
    for path, file in zip(fpaths, files):
        with open(path) as f:
            contents = f.readlines()
            pst = os.path.split(path)
            print(f'{file.ljust(10)}{contents[2].strip()}')


def create_backups(fpath):
    fpaths, files = get_files(fpath)
    if not os.path.isdir(fpath + '/BACKUPS'):
        print('Creating backups.')
        os.mkdir(fpath + '/BACKUPS')
        for path in fpaths:
            shutil.copy(path, os.path.split(path)[0] + '/BACKUPS/' + os.path.split(path)[1])
    else:
        print('Backups already found.')


def cleanup(fpath):
    """Turns turns apostrophe-separated into comma-separated
     and comma-decimal CSV into dot-decimal CSV."""
    fpaths, files = get_files(fpath)
    # Create backups
    create_backups(fpath)

    # Correct data
    for i in range(len(fpaths)):
        with open(fpaths[i], 'r') as read:
            lines = read.readlines()
        # Comma-separated regex: decimal . decimals e+ 2decimals, possible minus (and then the same again)
        pattern = re.compile(r'[-]?\d\.\d+[e][+-]\d\d[,][-]?\d\.\d+[e][+-]\d\d')
        matched = pattern.search(lines[0])

        # Fix delimiters
        if not matched == None:
            lines = [line.replace(',', ';') for line in lines]

        # Fix decimals
        lines = [line.replace(',', '.') for line in lines]

        with open(fpaths[i], 'w') as write:
            write.writelines(lines)


def load_samples_in_set(fpath):
    samples = []
    fpaths, files = get_files(fpath)
    print('Loading {0} files...'.format(len(files)))
    for path in fpaths:
        file = pd.read_csv(path, delimiter=',|;', engine='python')  # python allows two separators with "or" (|)
        file = np.array(file)
        WLV = file[:, 0]
        intensities = file[:, 1]
        name = os.path.split(path)[1].split('.')[0]
        samples.append(HSI.Spectra(intensities, WLV, name))
    return samples


def check_compatibility(fpath):
    set = load_samples_in_set(fpath)
    wlv_specs = []
    for spec in set:
        current = (len(spec.wlv), min(spec.wlv), max(spec.wlv))
        if current not in wlv_specs:
            wlv_specs.append(current)
    if len(wlv_specs) == 1:
        print('All WLVs have the same specs (len, m9in, max): {}'.format(wlv_specs[0]))
    else:
        print('WLV lengths differ (len, min, max):')
        [print('Len: {0} | Min: {1} | Max: {2} | Avg. bin size: {3}'
               .format(i[0], i[1], i[2], ((i[2] - i[1]) / i[0]))) for i in wlv_specs]

    print('{0} unique WLVs detected.'.format(len(wlv_specs)))