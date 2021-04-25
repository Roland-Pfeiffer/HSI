#!/usr/bin/env python3
import numpy as np

a = np.array([1, 2, 3])
print(a.ndim)

if a.ndim == 1:
    a = np.array([a, ])

print(a.ndim)
print(a)