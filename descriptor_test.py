#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import HSI
import load_sample_spectra

# Pseudo spectrum and descriptor
pseudo_spectrum = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.75, 2.0, 3.4, 5.0, 4.9, 4.1, 2.3, 0.1, 0.1, 0.1])
pseudo_wlv = np.array(list(range(1, 16)))
desc_start_index = 4
desc_stop_index = 12
ls_start = np.linspace(0, 1, 9 - (5 - 1))
ls_stop = np.linspace(1, 0, 13 - (9 - 1))
descplot = np.hstack([np.repeat([np.nan], 4), ls_start, ls_stop[1:], np.repeat([np.nan], 2)])
print(descplot)
print(len(pseudo_spectrum))
print(len(descplot))
plt.plot(pseudo_wlv, np.array([pseudo_spectrum, descplot]).T)
plt.show()
# Cut spectrum to descriptor
pseudo_spectrum_section = pseudo_spectrum[desc_start_index:desc_stop_index + 1]
desc_merged = np.hstack([ls_start, ls_stop[1:]])
print(len(pseudo_spectrum_section))
print(len(desc_merged))

# # Actual spectrum and descriptor
# data, wavelengths, mat_names = load_sample_spectra.load_spectra()
# wlv = HSI.WavelengthVector(wavelengths)
# spectrum = HSI.Spectra(data, wlv, mat_names)
# test_desc = HSI.TriangleDescriptor('testmat', 3143, 3303, 3399, wlv)
# test_desc.compare_to_spectrum(spectrum)