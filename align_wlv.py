#!/usr/bin/env python3
import numpy as np
from scipy.signal import correlate
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]\t%(message)s')


def align_wlv(a, b):
    """Aligns WLV b with WLV a.
    ToDo: Add threshold?
    ToDo: Offer UNION, INTERSECT, ORIGINAL and NEW
    ToDo: align spectra!
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


a = np.array([0, 1, 2, 3])
b = np.array([-0.9, 0.1, 1.1, 2.1, 3.1])
ba = align_wlv(a, b)
print(a)
print(ba)
print()

a = np.array([-0.9, 0.1, 1.1, 2.1, 3.1])
b = np.array([0, 1, 2, 3])
ba = align_wlv(a, b)
print(a)
print(ba)
