#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import HSI
import load_sample_spectra
# import logging
# logging.basicConfig(level=logging.DEBUG, format='[ %(levelname)s ]\t%(message)s')

# Get pearsons r:
test_descriptor = HSI.TriangleDescriptor(4, 9, 13, "Fake")
desc_perf = test_descriptor.compare_to_spectrum(fake_spec.intensities, fake_spec.wlv)
# print(desc_perf)

# Actual spectrum and descriptor
data, wlv, mat_names = load_sample_spectra.load_spectra()
spectrum = HSI.Spectra(data, wlv, mat_names)
# test_desc = HSI.TriangleDescriptor('testmat', 3143, 3303, 3399, wlv)
# test_desc.compare_to_spectrum(spectrum)

print(wlv)
print(data)
print(data.shape)