import HSI
import numpy as np
import spectral
import matplotlib.pyplot as plt

fname = "/media/findux/DATA/HSI_Data/imec_sample data/sample_data_pills_and_leaf.hdr"

hdr = spectral.open_image(fname)
print(hdr.shape)
# Avoid reading the entire cube:
spectrum = hdr.read_pixel(700, 700)

# plt.style.use("dark_background")

