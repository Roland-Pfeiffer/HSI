#!/usr/bin/env python3

import os
import re

fpath = '/media/findux/DATA/HSI_Data/reference_spectra_josef/'
files = sorted(os.listdir(fpath))
fpaths = [fpath + file for file in files]
material_names = [name.split('.')[0] for name in files]

print(fpaths)
print(files)

for i in range(len(fpaths)):
    with open(fpaths[i], 'r') as read:
        lines = read.readlines()


    # Comma-separated regex: decimal . decimals e+ 2decimals
    pattern = re.compile(r'\d\.\d+[e][+-]\d\d[,]\d\.\d+[e][+-]\d\d')
    matched = pattern.search(lines[0])
    print(matched)
    if not matched == None:
        lines = [line.replace(',', ';') for line in lines]

    lines = [line.replace(',', '.') for line in lines]


    with open(fpaths[i], 'w') as write:
        write.writelines(lines)
