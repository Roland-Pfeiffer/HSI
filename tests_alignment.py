#!/usr/bin/env python3
import numpy as np
import HSI
import pandas as pd
import time
import matplotlib.pyplot as plt
import create_alignment_test_data
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t%(message)s')
logging.disable()


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


# data = create_alignment_test_data.create_data()
# wlmin = np.min([np.min(spc.wlv) for spc in data])
# wlmax = np.max([np.max(spc.wlv) for spc in data])
#
# plt.figure()
# for spec in data:
#     print(spec.wlv)
#     plt.plot(spec.wlv, spec.intensities.T)
#     plt.xlim(wlmin, wlmax)
# plt.draw()
#
#
#
# # Align:
# # This only works with WLVs of the same resolution. If they have a different resolution, you need to interpolate.
# # ToDo: Find a workaround for different resolutions. Probably: interpolate.
#
# # Verify resolution is the same
# resolutions = np.array([spc.wlv[1] - spc.wlv[0] for spc in data])
# print(resolutions)
# same = np.all(resolutions == resolutions[0])
# if not same:
#     raise ValueError('Wavelength vectors do not have the same resolution.\n'
#                      'Interpolation is not implemented yet.')
#
# final_spectra = HSI.Spectra(data[0].intensities, data[0].wlv, data[0].material)
# for spec in data[1:]:
#     wlv_min = min(spec.wlv)
#     wlv_max = max(spec.wlv)
#     wlv_aligned = HSI.align_wlv(spec.wlv, final_spectra.wlv)
#     spec.wlv = wlv_aligned
#     # Detect overlap possibilities and cut accordingly:
#     # Cut overhanging parts of previously added spectra if they overhang the new WLV
#     # Cut the lower end
#     if min(final_spectra.wlv) < wlv_min:
#         logging.debug(np.where(final_spectra.wlv == min(spec.wlv))[0])
#         start_i = int(np.where(final_spectra.wlv == min(spec.wlv))[0])
#         final_spectra.intensities = final_spectra.intensities[:, start_i:]
#         final_spectra.wlv = final_spectra.wlv[start_i:]
#     # Cut the upper end
#     if max(final_spectra.wlv) > wlv_max:
#         logging.debug(np.where(final_spectra.wlv == max(spec.wlv))[0])
#         stop_i = int(np.where(final_spectra.wlv == max(spec.wlv))[0] + 1)
#         final_spectra.intensities = final_spectra.intensities[:, :stop_i]
#         final_spectra.wlv = final_spectra.wlv[:stop_i]
#     # Cut the new spectra to shape
#     # Cut lower end
#     if spec.wlv[0] == spec.wlv[1]:
#         logging.debug(np.where(spec.wlv == min(final_spectra.wlv))[0])
#         start_i = max(np.where(spec.wlv == min(final_spectra.wlv)[0]))  # Take highest, bc. there are "leading min()s"
#         spec.intensities = spec.intensities[:, start_i:]
#         spec.wlv = spec.wlv[start_i:]
#     # Cut upper end
#     if spec.wlv[-1] == spec.wlv[-2]:
#         logging.debug(np.where(spec.wlv == max(final_spectra.wlv))[0])
#         stop_i = min(np.where(spec.wlv == max(final_spectra.wlv))[0]) + 1  #  Take lowest: "trailing max()s"
#         spec.intensities = spec.intensities[:, :stop_i]
#         spec.wlv = spec.wlv[:stop_i]
#
#     # Append new spectra
#     final_spectra.add_spectra(spec)
#
# print(final_spectra.wlv)
# plt.figure()
# final_spectra.plot()
# plt.xlim(wlmin, wlmax)
# plt.show()



data = create_alignment_test_data.create_data()

def crop_spectra(Spectra_list):
    """Takes a list of spectra objects, aligns their wavelength vectors and crops them to the overlapping region, if
    there is an overhang on either side.
    NOTE: This requires the WLVs to have the same resolution i.e. bin size."""
    # Align:
    # This only works with WLVs of the same resolution. If they have a different resolution, you need to interpolate.
    # ToDo: Find a workaround for different resolutions. Probably: interpolate.

    # Verify resolution is the same
    resolutions = np.array([spc.wlv[1] - spc.wlv[0] for spc in Spectra_list])
    print(f'Resolutions: {resolutions}')
    same = np.all(resolutions == resolutions[0])
    if not same:
        raise ValueError('Wavelength vectors do not have the same resolution.\n'
                         'Interpolation is not implemented yet.')  # ToDo: Remove the raise error once resolved.

    # First, just copy the first spectrum:
    final_spectra = HSI.Spectra(Spectra_list[0].intensities, Spectra_list[0].wlv, Spectra_list[0].material)
    for spec in data[1:]:
        wlv_min = min(spec.wlv)
        wlv_max = max(spec.wlv)
        wlv_aligned = HSI.align_wlv(spec.wlv, final_spectra.wlv)
        spec.wlv = wlv_aligned
        # Detect overlap possibilities and cut accordingly:
        # Cut overhanging parts of previously added spectra if they overhang the new WLV
        # Cut the lower end
        if min(final_spectra.wlv) < wlv_min:
            logging.debug(np.where(final_spectra.wlv == min(spec.wlv))[0])
            start_i = int(np.where(final_spectra.wlv == min(spec.wlv))[0])
            final_spectra.intensities = final_spectra.intensities[:, start_i:]
            final_spectra.wlv = final_spectra.wlv[start_i:]
        # Cut the upper end
        if max(final_spectra.wlv) > wlv_max:
            logging.debug(np.where(final_spectra.wlv == max(spec.wlv))[0])
            stop_i = int(np.where(final_spectra.wlv == max(spec.wlv))[0] + 1)
            final_spectra.intensities = final_spectra.intensities[:, :stop_i]
            final_spectra.wlv = final_spectra.wlv[:stop_i]
        # Cut the new spectra to shape
        # Cut lower end
        if spec.wlv[0] == spec.wlv[1]:
            logging.debug(np.where(spec.wlv == min(final_spectra.wlv))[0])
            start_i = max(np.where(spec.wlv == min(final_spectra.wlv)[0]))  # Take highest, bc. there are "leading min()s"
            spec.intensities = spec.intensities[:, start_i:]
            spec.wlv = spec.wlv[start_i:]
        # Cut upper end
        if spec.wlv[-1] == spec.wlv[-2]:
            logging.debug(np.where(spec.wlv == max(final_spectra.wlv))[0])
            stop_i = min(np.where(spec.wlv == max(final_spectra.wlv))[0]) + 1  #  Take lowest: "trailing max()s"
            spec.intensities = spec.intensities[:, :stop_i]
            spec.wlv = spec.wlv[:stop_i]

        # Append new spectra
        final_spectra.add_spectra(spec)
    return final_spectra


raw_spec = create_alignment_test_data.create_data()



xmin, xmax = np.min([np.min(sp.wlv) for sp in raw_spec]), np.max([np.max(sp.wlv) for sp in raw_spec])

plt.figure()
for spc in raw_spec:
    plt.plot(spc.wlv, spc.intensities.T)
plt.xlim(xmin, xmax)
plt.draw()

cropped_spec = crop_spectra(raw_spec)

plt.figure()
cropped_spec.plot()
plt.xlim(xmin, xmax)
plt.show()





# # Might be interesting for aligning too:
# wlv_1 = data[0].wlv
# wlv_2 = data[1].wlv
# print(wlv_1)
# print(wlv_2)
# wlv_2 = HSI.align_wlv(wlv_2, wlv_1)
# print(wlv_2)
# print(np.intersect1d(wlv_1, wlv_2))
