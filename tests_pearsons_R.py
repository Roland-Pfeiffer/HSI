#!/usr/bin/env python3
import numpy as np
from scipy.stats import pearsonr

a = np.array([1, 2, 3, 6, 5, 4, 7, 8, 9])
b = np.array([1, 2, 3, 6, 5, 4, 7, 8, 10])

c = np.sort(a)
d = np.sort(b)

first = pearsonr(a, b)
second = pearsonr(c, d)

print(first)
print(second)

print(f'Identical: {first == second}')