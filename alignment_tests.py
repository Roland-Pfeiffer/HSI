#!/usr/bin/env python3
import numpy as np
import HSI
import pandas as pd
import time


pd.options.display.width = 0  # To make pandas autodetect terminal width


# Speed comparisons
fpath_PE = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PE.csv'
fpath_PHB = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PHB.CSV'
pe = pd.read_csv(fpath_PE, sep=';')
phb = pd.read_csv(fpath_PHB, sep=';')
pe_wlv = np.array(pe.iloc[:, 0])
phb_wlv = np.array(phb.iloc[:, 0])


# # Avg. over 1000 iterations
# floops = []
# lcomps = []
# for i in range(1000):
#     if i%25 == 0: print(f'Iteration {i}')
#     t0 = time.perf_counter()
#     c = HSI.align_wlv(pe_wlv, phb_wlv)
#     t1 = time.perf_counter()
#     floops.append(t1 - t0)
#
#     t0 = time.perf_counter()
#     c = HSI.align_wlv_fast(pe_wlv, phb_wlv)
#     t1 = time.perf_counter()
#     lcomps.append(t1 - t0)
# print(np.mean(floops))
# print(np.mean(lcomps))