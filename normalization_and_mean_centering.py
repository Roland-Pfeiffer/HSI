import HSI
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

fname = '/media/findux/DATA/HSI_Data/SPECIM_field_data/2019-06-04_Flatskär/2019-06-04_005/results/REFLECTANCE_2019-06-04_005.hdr'
fname = '/media/findux/DATA/HSI_Data/SPECIM_field_data/2019-06-04_Flatskär/2019-06-04_006/capture/2019-06-04_006.hdr'

print('Loading HSI...')
hdr, cube, wlv = HSI.load_hsi(fname)
print('Cube shape: {}'.format(hdr.shape))
spectra = HSI.unfold_cube(cube)
spectra = HSI.Spectra(spectra, wlv)
print(spectra.intensities.shape)
print('Min: {0} | Max: {1} | Mean: {2}'.format(np.min(spectra.intensities),
                                               np.max(spectra.intensities),
                                               np.mean(spectra.intensities)))
spectra = spectra.random_subsample(10000)

# maxval = np.max(spectra.intensities)
# print(maxval)
# threshold_max = 1000
# threshold_min = 100
# out_of_range = (spectra.intensities < threshold_min) | (spectra.intensities > threshold_max)
# in_vector = [not np.any(row) for row in out_of_range]
# print('{} samples kept in.'.format(in_vector.count(True)))
# spectra.intensities = spectra.intensities[in_vector, :]
# print(spectra.intensities.shape)

# plt.figure('Histogram')
# plt.hist(spectra.intensities.flatten())
# plt.title('Histogram of intensities')

# for f in range(len(spectra.wlv)):
#     feature = spectra.intensities[:, f]
#     sns.distplot(feature)
# plt.show()

# plt.figure('Original')
# plt.plot(wlv, spectra.intensities.T)
#
# print('Scaling...')
# spectra_scaled = HSI.Spectra(preprocessing.scale(spectra.intensities), wlv)
# plt.figure('Scaled')
# plt.plot(wlv, spectra_scaled.intensities.T)
#
# print('Normalizing...')
# spectra_scaled_norm = HSI.Spectra(preprocessing.normalize(spectra_scaled.intensities), wlv)
# plt.figure('Scaled and normalized')
# plt.plot(wlv, spectra_scaled_norm.intensities.T)
# plt.show()

# Accounting for the lognormal distribution of the values:
# ToDo: This does not work yet
power_transform = preprocessing.PowerTransformer(method='box-cox', standardize=False)
spectra_transformed = spectra
spectra_transformed.intensities = power_transform.fit_transform(spectra_transformed.intensities)
