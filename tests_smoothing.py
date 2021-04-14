import HSI
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

px_leaf = HSI.load_pixel(fname, 700, 700, 'Leaf')
px_pill = HSI.load_pixel(fname, 1130, 1140, 'Pill')

spects = px_leaf
spects.add_spectra(px_pill)

plt.figure('Original')
plt.plot(spects.wlv, spects.intensities.T)
plt.title('Original')

spects_smoothed = HSI.Spectra(savgol_filter(spects.intensities, 15, 2), spects.wlv)

plt.figure('Smoothed')
plt.plot(spects_smoothed.wlv, spects_smoothed.intensities.T)
plt.title('Smoothed')

smooth_derivative = HSI.Spectra(savgol_filter(spects.intensities, 15, 2, 1), spects.wlv)

plt.figure('Smoothed derivative')
plt.plot(smooth_derivative.wlv, smooth_derivative.intensities.T)
plt.title('Smoothed derivative')

plt.show()