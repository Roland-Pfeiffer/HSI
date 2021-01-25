#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import HSI
import load_sample_spectra
# import logging
# logging.basicConfig(level=logging.DEBUG, format='[ %(levelname)s ]\t%(message)s')

# Pseudo spectrum and descriptor
pseudo_spectrum = np.array([0.1, 0.1, 2.3, 0.1, 0.1, 0.75, 2.0, 3.4, 5.0, 4.9, 4.1, 2.3, 0.1, 0.1, 0.1])
pseudo_wlv = np.array(list(range(1, 16)))

# print(pseudo_spectrum)
# print(pseudo_wlv)
# plt.figure('Spectrum intensities')
# plt.plot(pseudo_wlv, pseudo_spectrum)
# plt.vlines(9, min(pseudo_spectrum), max(pseudo_spectrum), linestyles='--', colors='red')
# plt.show()


fake_spec = HSI.Spectra(pseudo_spectrum, pseudo_wlv)



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