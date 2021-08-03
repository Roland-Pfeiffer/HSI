#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import HSI

ints = np.array([[1, 1, 1, 2, 2, 3, 2, 1, 1],
                 [1, 2, 3, 4, 3, 2, 1, 1, 1],
                 [4, 3, 2, 2, 1, 1, 2, 1, 1]])
wlv = np.array([110, 111, 112, 113, 114, 115, 116, 117, 118])
mats = ['Leaf', 'Pill', 'Pill']

data = HSI.Spectra(ints, wlv, mats)
data.plot()

print(repr(data))