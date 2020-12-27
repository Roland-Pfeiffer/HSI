import HSI
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter
from sklearn import preprocessing
import spectral
from time import perf_counter as pfc
import matplotlib.pyplot as plt
from PIL import Image
import load_sample_spectra

# fname = '/media/findux/DATA/HSI_Data/recycling, sorting/white_plastics_mask.png'
#
# mask_img = Image.open(fname)
# mask_img = np.array(mask_img)
# print(mask_img.shape)
#
# mask_img = plt.imread(fname)[:, :, 0]


# mask_layer = np.array(mask_img[:, :, 0])
# print(np.max(mask_layer))
# x, y = mask_layer.shape
# mask_vector = mask_layer.reshape(x * y)
# mask_vector_i = np.where(mask_vector == 1)[0]
# print(mask_vector_i)


# Creating linspaces
ls = np.linspace(0, 1, 10)
print(ls)