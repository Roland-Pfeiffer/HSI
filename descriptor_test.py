#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import random
import HSI
import load_sample_spectra
import logging
logging.basicConfig(level=logging.DEBUG, format='[ %(levelname)s ]\t%(message)s')

# Actual spectrum and descriptor
data, wlv, mat_names = load_sample_spectra.load_spectra()
spectrum = HSI.Spectra(data, wlv, mat_names)

sample_spectrum = spectrum.random_subsample(16, seed=42)
print(sample_spectrum.intensities.shape)
plt.plot(sample_spectrum.wlv, sample_spectrum.intensities.T)
plt.show()

d1 = HSI.TriangleDescriptor(2792, 2983, 3143, 'test')
d2 = HSI.TriangleDescriptor(1609, 1745, 1801, 'Test')
d3 = HSI.TriangleDescriptor(680, 700, 720, 'test')
set = HSI.DescriptorSet(d1)
set.add_descriptor(d2)
set.add_descriptor(d3)

print(set)

mat = set.correlate(sample_spectrum.intensities, sample_spectrum.wlv)
print(mat)