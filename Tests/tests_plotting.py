from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter
from sklearn import preprocessing
import matplotlib.pyplot as plt
import load_sample_spectra
import HSI
import sys
import random
import numpy as np
import seaborn as sns
import pandas as pd


# # Using Josef's sample spectra:
# data, wlv, material_names = load_sample_spectra.load_spectra()
# print(material_names)
# PA6 = data[:, 0]
# range_PA6 = max(PA6) - min(PA6)
# print(PA6.shape)
# print('Extracting peaks...')
# peaks = find_peaks(PA6, prominence=range_PA6 * 0.15)  # NOTE: Gives funky results with NaNs
# peak_proms = peak_prominences(PA6, peaks[0])
# peak_i = peaks[0]
# # plt.plot(wlv, PA6)
# plt.vlines(wlv[peak_i], ymin=min(PA6), ymax=max(PA6), colors='Yellow')
# # Show raw spectrum
# # plt.figure('Raw')
# # plt.plot(wlv, data)
# # Scale and normalize spectrum:
# data_scaled = preprocessing.scale(data)
# print('Scaled: {}'.format(data_scaled.shape))
# data_scaled_norm = preprocessing.normalize(data)
# print('Scaled and normed: {}'.format(data_scaled_norm.shape))
# # plt.figure('Scaled and Normalized)')
# # plt.plot(wlv, data_scaled_norm)


# Use IMEC:
fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'
fname_mask = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf_MASK.png'

# Using ???
fname = '/media/findux/DATA/HSI_Data/recycling, sorting/white_plastics.hdr'
fname_mask = '/media/findux/DATA/HSI_Data/recycling, sorting/white_plastics_mask.png'

# Using Specim:
fname = '/media/findux/DATA/HSI_Data/SPECIM_field_data/2019-06-04_Flatskär/2019-06-04_005/capture/2019-06-04_005.hdr'

# Josef's strange spectra
fname ='/home/findux/Desktop/HSI GU/10x_PET Particle_corrected.hdr'

print('Loading hdr...')
hdr, cube, wlv = HSI.load_hsi(fname)
print(type(cube), cube.shape)
specs_unf = HSI.unfold_cube(cube)
print(type(specs_unf), specs_unf.shape)
spectra = HSI.Spectra(specs_unf, wlv)

specs_subsample = spectra.random_subsample(10000)

plt.figure('Raw')
plt.plot(spectra.wlv, spectra.intensities.T)
# print('Scaling...')
# spectra_scaled = HSI.Spectra(preprocessing.scale(spectra.intensities), wlv)
# # plt.figure('Scaled')
# # plt.plot(wlv, spectra_scaled.intensities.T)
#
# print('Normalizing...')
# spectra_scaled_norm = HSI.Spectra(preprocessing.normalize(spectra_scaled.intensities), wlv)
# plt.figure('Scaled and normalized')
# plt.plot(wlv, spectra_scaled_norm.intensities.T)
# plt.show()

# # Seaborn
# # AINT NOBODY GOT TIME FOR THIS
# print(wlv.shape)
# print(spectra_scaled_norm.intensities.shape)
# merged = np.vstack((wlv.T, spectra_scaled_norm.intensities))
# merged = pd.DataFrame(merged.T)
# merged.rename(columns={0: 'WLV'}, inplace=True)
# print(merged)
# g = sns.lineplot(data=merged)
# g.plot()
# plt.show()
# # g = sns.catplot(x="X_Axis", y="vals", hue='cols', data=merged_melted)
