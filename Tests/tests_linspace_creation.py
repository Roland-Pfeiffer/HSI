#!/usr/bin/env python3
import numpy as np

wlv = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
start_i = 3
peak_i = 7
stop_i = 11

before_peak_len = peak_i - start_i
after_peak_len = stop_i - peak_i + 1

last_value_before_peak = 1 - (1 / before_peak_len)
print(last_value_before_peak)
asc_linspace = np.linspace(0, last_value_before_peak, before_peak_len)
desc_linspace = np.linspace(1, 0, after_peak_len)
print(asc_linspace)
print(desc_linspace)

triangle_array = np.hstack([asc_linspace, desc_linspace])
print(triangle_array)