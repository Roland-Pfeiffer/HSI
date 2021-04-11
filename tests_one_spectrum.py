import HSI
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

# print('Loading HSI data...')
# hdr, cube, wlv = HSI.load_hsi(fname)
# # Plot img
# fname_png = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.png'
# png = plt.imread(fname_png)
# plt.imshow(png)
# plt.show()
# print('Unfolding...')
# spectra = HSI.Spectra(HSI.unfold_cube(cube), wlv)
# print('Select subsample')
# spectra_sample = spectra.random_subsample().intensities
# plt.style.use('dark_background')
# plt.plot(spectra.wlv, spectra_sample.T)
# plt.show()


px_leaf = HSI.load_pixel(fname, 700, 700, 'Leaf')
px_pill = HSI.load_pixel(fname, 1130, 1140, 'Pill')

spects = px_leaf
spects.add_spectra(px_pill)

plt.figure('Original')
plt.plot(spects.wlv, spects.intensities.T)
plt.title('Original')

derivs = [np.gradient(spects.intensities[i, :]) for i in range(spects.intensities.shape[0])]
derivs = np.array(derivs)

plt.figure('1st derivative')
plt.plot(spects.wlv, derivs.T)
plt.title('1st derivative')

spects_smoothed = HSI.Spectra(savgol_filter(spects.intensities, 15, 2), spects.wlv)

# plt.figure('Smoothed')
# plt.plot(spects_smoothed.wlv, spects_smoothed.intensities.T)
# plt.title('Smoothed')

smooth_derivs = [np.gradient(spects_smoothed.intensities[i, :]) for i in range(spects_smoothed.intensities.shape[0])]
smooth_derivs = np.array(smooth_derivs)

plt.figure('Smoothed')
plt.plot(spects.wlv, smooth_derivs.T)
plt.title('Smoothed')

plt.show()
