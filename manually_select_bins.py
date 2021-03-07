#!/usr/bin/env python3

import matplotlib.pyplot as plt
import load_sample_spectra
import HSI
from sklearn.preprocessing import normalize

import numpy as np
# import logging
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s]\t%(message)s')

files = load_sample_spectra.load_samples_in_set('/media/findux/DATA/HSI_Data/reference_spectra_josef/')
print(files)
del files[1]  # Drop the one with the different wlv
# ToDo: Once the alignment/overlap script is done, include this spectrum again
print(files)
for file in files:
    print(len(file.wlv))

# Merge files into one spectrum class:
spectra = HSI.Spectra(files[0].intensities, files[0].wlv, files[0].material_column)
for spec in files[1:]:
    spectra.add_spectra(spec)
spectra.plot()
plt.show()
for file in files:
    print(file.material_column)

# Select three values (for tri_desc) by hand

spec = files[0]
print('Spectrum:')
print(spec.intensities)
print(type(spec.intensities))
material = spec.material_column[0]


# Obtain descriptor positions:
descs = []
while True:
    plt.plot(spec.wlv, spec.intensities)
    plt.title(spec.material_column)
    coords = plt.ginput(3)
    plt.close()
    if len(coords) == 3:
        coords = [coord[0] for coord in coords]
        print(f'Start: {coords[0]} | Peak: {coords[1]} | Stop: {coords[2]}')
        descs.append(tuple(coords))
    else:
        print('Using preset descriptors')
        descs = [(2890.826381115591, 2933.414111223118, 3005.2809057795703),
                 (3135.705829233871, 3295.409817137097, 3423.1730074596776),
                 (1490.7547538306453, 1562.621548387097, 1610.5327447580644),
                 (1586.5771465725807, 1658.4439411290323, 1770.2367326612903),
                 (1937.9259199596777, 2193.452300604839, 2504.875077016129)]
        break

    cont_UI = input('Add more points? (type "n" for no)\n >> ')
    if cont_UI.lower() == 'n':
        break

descs = [HSI.TriangleDescriptor(d[0], d[1], d[2], material) for d in descs]

# Create desc set
for i, desc in enumerate(descs):
    if i == 0:
        set = HSI.DescriptorSet(desc)
    else:
        set.add_descriptor(desc)
print(set)

corr_mat = set.correlate(spec.intensities, spec.wlv)
print(corr_mat)