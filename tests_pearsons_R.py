#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

a = np.array([1, 2, 3, 6, 5, 4, 7, 8, 9])
b = np.array([1, 2, 3, 6, 5, 4, 7, 8, 10])
x = np.arange(0, len(a))

plt.plot(x, a, alpha=0.5)
plt.plot(x, b, alpha=0.5)
plt.show()


c = np.sort(a)
d = np.sort(b)

first = pearsonr(a, b)
second = pearsonr(c, d)

print(first)
print(second)

print(f'Identical: {first == second}')

a = np.array([1, 2, 3, 6, 5, 4, 7, 8, 9])
b = np.array([1, 2, 5, 6, 3, 4, 7, 8, 10])
plt.plot(x, a, alpha=0.5)
plt.plot(x, b, alpha=0.5)
plt.show()
print(pearsonr(a, b))


