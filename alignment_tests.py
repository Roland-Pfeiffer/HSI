#!/usr/bin/env python3
import numpy as np
import HSI
import pandas as pd
import time
import matplotlib.pyplot as plt
import create_alignment_test_data



# pd.options.display.width = 0  # To make pandas autodetect terminal width
# # Speed comparisons
# fpath_PE = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PE.csv'
# fpath_PHB = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PHB.CSV'
# pe = pd.read_csv(fpath_PE, sep=';')
# phb = pd.read_csv(fpath_PHB, sep=';')
# pe_wlv = np.array(pe.iloc[:, 0])
# phb_wlv = np.array(phb.iloc[:, 0])

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


data = create_alignment_test_data.create_data()
for spec in data:
    print(spec.wlv)
    plt.plot(spec.wlv, spec.intensities.T)
plt.show()

# Align:
# ToDo: This only works with WLVs of the same resolution. If they have a different resolution, you need to interpolate.
final_spectra = HSI.Spectra(data[0].intensities, data[0].wlv, data[0].material_column)
for spec in data[1:]:
    wlv = spec.wlv
    ints = spec.intensities
    names = spec.material_column
    # Align WLV
    wlv_aligned = HSI.align_wlv(wlv, final_spectra.wlv)

    # Find where overlap starts
    wlv_start = 0
    for i in range(len(wlv_aligned)):
        if wlv_aligned[i] == wlv_aligned[i+1]:
            wlv_start += 1
        else:
            break
    # Find where overlap stops
    wlv_stop = len(wlv_aligned) + 1
    for i in range(len(wlv_aligned)).__reversed__():
        if wlv_aligned[i] == wlv_aligned[i - 1]:
            wlv_stop -= 1
        else:
            break

    # Cut new data to shape
    wlv_cut = spec.wlv[wlv_start:wlv_stop]  # Because [:-0] does not work
    ints_cut = spec.intensities[:, wlv_start:wlv_stop]  # Because [:-0] does not work

    print(final_spectra.wlv)
    print(min(wlv_cut))

    # Cut old data to shape if it overhangs the new WLV
    # Left cut off:
    if min(final_spectra.wlv) < min(wlv_cut):
        start_i = int(np.where(final_spectra.wlv == min(wlv_cut))[0][0])
        final_spectra.intensities = final_spectra.intensities[:, start_i:]
    # Right cutoff:
    if max(final_spectra.wlv) > max(wlv_cut):
        stop_i = int(np.where(final_spectra.wlv == max(wlv_cut))) + 1
        final_spectra.intensities = final_spectra.intensities[:, :stop_i]












# # Might be interesting for aligning too:
# wlv_1 = data[0].wlv
# wlv_2 = data[1].wlv
# print(wlv_1)
# print(wlv_2)
# wlv_2 = HSI.align_wlv(wlv_2, wlv_1)
# print(wlv_2)
# print(np.intersect1d(wlv_1, wlv_2))
