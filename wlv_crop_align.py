#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
import logging

import create_test_data
import HSI

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t%(message)s')
# logging.disable()


def align_wlv(a, b):
    """Aligns WLV b with WLV a. NOTE: This only shifts vector b as little as possible so that their values overlap
    it does not fill them.
    ToDo: Add threshold?
    ToDo: Offer UNION, INTERSECT, ORIGINAL and NEW
    ToDo: align spectra! This needs to carry over how many bins have been cut off from each side.
    """
    if (len(a) == len(b)) and np.all(a == b):
        logging.info('WLVs are identical.')
        # return a

    # Test if WLV intervals are same:
    a_ints = [a[i+1] - a[i] for i in range(len(a)-1)]
    b_ints = [b[i+1] - b[i] for i in range(len(b)-1)]
    if not np.unique(a_ints) == np.unique(b_ints):
        raise AssertionError('WLVs do not have the same interval.')
    interval = int(np.unique(a_ints))

    # Alignment
    b_aligned = []
    # If only the end differs
    if min(a) == min(b):
        pass  # ToDo
    # If the start differs:
    if min(a) < min(b):
        closest_a_i = int(np.argmin(abs(min(b) - a)))
        logging.debug('Element 0 in b aligns closest with element {} in a.'.format(closest_a_i))
        b[0] = a[closest_a_i]
        # Copy values if possible:
        if len(b) == len(a[closest_a_i:]):
            b_aligned = a[closest_a_i:]
        else:
            for i in range(len(b)):
                b_aligned[i] = a[closest_a_i] + interval * i
    elif min(a) > min(b):
        logging.debug('Element 0 in a aligns closest with element {} in b.'.format(np.argmin(abs(min(a) - b))))
        closest_b_i = int(np.argmin(abs(min(a) - b)))
        b_aligned = [a[0] - interval * (i + 1) for i in range(closest_b_i).__reversed__()]
        for i in range(len(b) - closest_b_i):
            b_aligned.append(interval * i)
    return b_aligned


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


if __name__ == '__main__':

    # ------------------------------------------------------------ Aligning --------------------------------------------
    a = np.array([0, 1, 2, 3])
    b = np.array([-0.9, 0.1, 1.1, 2.1, 3.1])
    ba = align_wlv(a, b)
    print(a)
    print(b)
    print(ba)
    print()

    a = np.array([-0.9, 0.1, 1.1, 2.1, 3.1])
    b = np.array([0, 1, 2, 3])
    ba = align_wlv(a, b)
    print(a)
    print(b)
    print(ba)

    # ------------------------------------------------------------- Cropping -------------------------------------------
    data = create_test_data.create_alignment_test_data()
    raw_spec = create_test_data.create_alignment_test_data()
    for spec in raw_spec:
        print(spec.wlv)
        print(spec.intensities)

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