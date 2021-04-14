import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t%(message)s')

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import HSI


fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'
desc_1 = HSI.TriangleDescriptor(669, 710, 769, 'T1')
# desc_2 = HSI.TriangleDescriptor

triplot = np.array([[desc_1.start_wl, desc_1.peak_wl, desc_1.stop_wl],
                    [0, 0.03, 0]])

a = np.array([1, 2, 3])
print(type(a))


px_leaf = HSI.load_pixel(fname, 700, 700, 'Leaf')
px_pill = HSI.load_pixel(fname, 1130, 1140, 'Pill')

spects = px_leaf
spects.add_spectra(px_pill)

print(type(spects.wlv))
print(type(spects.intensities))

spects = spects.smoothen_savgol(1, 0, 0)

# plt.figure('Original')
# plt.plot(spects.wlv, spects.intensities.T)
# plt.title('Original')

plt.figure('1st derivative')
plt.plot(spects.wlv, spects.intensities.T)
plt.title('1st derivative')
plt.show()

# plt.figure('Smoothed')
# plt.plot(spects.wlv, spects.intensities.T)
# plt.title('Smoothed')


# plt.figure('Smoothed')
# plt.title('Smoothed')
# plt.plot(spects.wlv, smooth_derivs.T)
# plt.plot(triplot[0], triplot[1])
#
# plt.plot()
# plt.show()
#
# set_1 = HSI.DescriptorSet(desc_1)
# corr_mat = set_1.correlate(spects.intensities, spects.wlv)
#
# print(corr_mat)