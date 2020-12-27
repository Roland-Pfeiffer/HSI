#!/usr/bin/env python3

import matplotlib.pyplot as plt
import load_sample_spectra

data, wlv, names = load_sample_spectra.load_spectra()

# Select three values (for tri_desc) by hand
plt.plot(wlv, data)
coords = plt.ginput(3)
x_coords = [xy[0] for xy in coords]
print(x_coords)