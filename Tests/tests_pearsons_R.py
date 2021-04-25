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

# a = np.hstack([np.linspace(0, 10, 100), np.linspace(10, 0, 100)])
# b = np.hstack([np.linspace(0, 10, 115), np.linspace(10, 0, 85)])
# c = np.hstack([np.linspace(0, 10, 120), np.linspace(10, 0, 80)])
# plt.plot(list(range(200)), a)
# plt.plot(list(range(200)), b)
# plt.plot(list(range(200)), c)
#
# plt.show()
#
# print(pearsonr(a, b))
# print(pearsonr(a, c))