#!/usr/bin/env python3

import spectral
import sklearn
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t%(message)s')
# logging.disable()

fname_specim = '/media/findux/DATA/HSI_Data/SPECIM_field_data/Sample_data/2017-08-15_004/capture/2017-08-15_004.hdr'
fname_imec = '/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr'

knn_classes = 3
knn_iterations = 100

# data_specim = HSI_import(fname_specim)
logging.info('Loading data...')
img = spectral.open_image(fname_imec)
img = img[1000:1800, 1000:1800, :]
print(img.shape)
logging.info('Loading array...')
logging.info('Starting kmeans...')
(m, c) = spectral.kmeans(img, knn_classes, knn_iterations)

print(m.shape)
print(c.shape)

# plt.figure()
# for i in range(c.shape[0]):
#     plt.plot(c[i])
# plt.grid()
# plt.show()

plt.figure('Classes')
plt.imshow(m)
plt.show()