import HSI
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

fname = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'
test_start = 669
test_peak = 710
test_stop = 769

triplot = np.array([[test_start, test_peak, test_stop],
                    [0, 0.03, 0]])

a = np.array([1, 2, 3])
print(type(a))


px_leaf = HSI.load_pixel(fname, 700, 700, 'Leaf')
px_pill = HSI.load_pixel(fname, 1130, 1140, 'Pill')

spects = px_leaf
spects.add_spectra(px_pill)

print(type(spects.wlv))
print(type(spects.intensities))

# plt.figure('Original')
# plt.plot(spects.wlv, spects.intensities.T)
# plt.title('Original')

derivs = [np.gradient(spects.intensities[i, :]) for i in range(spects.intensities.shape[0])]
derivs = np.array(derivs)

# plt.figure('1st derivative')
# plt.plot(spects.wlv, derivs.T)
# plt.title('1st derivative')

spects.smoothen(9, 2, 0)

# plt.figure('Smoothed')
# plt.plot(spects.wlv, spects.intensities.T)
# plt.title('Smoothed')

smooth_derivs = [np.gradient(spects.intensities[i, :]) for i in range(spects.intensities.shape[0])]
smooth_derivs = np.array(smooth_derivs)

plt.figure('Smoothed')
plt.title('Smoothed')
plt.plot(spects.wlv, smooth_derivs.T)
# plt.plot(triplot[0], triplot[1])
plt.plot()
plt.show()

set_1 = HSI.DescriptorSet(desc_1)
corr_mat = set_1.correlate(spects.intensities, spects.wlv)

print(corr_mat)