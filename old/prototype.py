#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import pyperspectral

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]\t%(message)s')
# logging.disable()

fname = '/media/findux/DATA/spectral_data/imec_sample data/sample_data_pills_and_leaf.hdr'

px_leaf = pyperspectral.load_pixel(fname, 700, 700, 'Leaf')
px_pill = pyperspectral.load_pixel(fname, 1130, 1140, 'Pill')
spects = px_leaf
spects.add_spectra(px_pill)

desc_1 = pyperspectral.TriangleDescriptor(669, 710, 769, 'T1')
desc_2 = pyperspectral.TriangleDescriptor(766, 783, 809, 'T1')
desc_3 = pyperspectral.TriangleDescriptor(810, 826, 850, 'T1')

set_1 = pyperspectral.DescriptorSet(desc_1)
set_1.add_descriptor(desc_2)
set_1.add_descriptor(desc_3)

# array to plot the triangle descriptor
triplot = np.array([[desc_1.start_wl, desc_1.peak_wl, desc_1.stop_wl],
                    [0, 0.03, 0]])

print(f'WLV vector type:  {type(spects.wlv)}')
print(f'Intensities type: {type(spects.intensities)} Shape: {spects.intensities.shape}')

spects = spects.smoothen_savgol(5, 0)
derivs = spects.turn_into_gradient()

print(spects.material)
print(spects.select_by_material('Leaf'))
print(spects.intensities)

# Double-axis plot code from: https://matplotlib.org/2.2.5/gallery/api/two_scales.html
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Original intensities', color=color)
ax1.plot(spects.wlv, spects.intensities.T, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Derivative', color=color)  # we already handled the x-label with ax1
ax2.plot(spects.wlv, derivs.intensities.T, color=color)
plt.plot(triplot[0], triplot[1], color='green')

ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

corr_mat = set_1.correlate(derivs.intensities, spects.wlv)
print(corr_mat)

spects.plot()
plt.show()