#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

wlv = np.array(list(range(400, 500, 8)))
spectum = np.random.standard_normal(13)

print(len(wlv), wlv)
print(spectum)

points = 415, 442, 465

indices = tuple([np.argmin(np.abs(wlv - point)) for point in points])
intensities = tuple([spectum[i] for i in indices])
print(type(indices), indices)


plt.plot(wlv, spectum, 'o--', label='Spectrum')
plt.plot(points, intensities, 'go', label='Original points')
plt.plot([wlv[i] for i in indices], intensities, 'rv', label='Binned points')
plt.grid()
plt.legend()
plt.show()