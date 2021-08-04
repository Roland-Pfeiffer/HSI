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

ints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 6])
data.add_spectra(HSI.Spectra(ints, wlv))
data.plot()

wlv_1 = np.array([0, 1, 2, 3, 4])
wlv_2 = np.array([2, 3, 4, 5])
wlv_aligned = HSI.align_wlv(wlv_1, wlv_2)
print(wlv_aligned)