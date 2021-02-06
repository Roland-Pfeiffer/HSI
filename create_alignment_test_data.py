#!/usr/bin/env python3
"""Create test data for alignment"""
import numpy as np
import HSI

def create_data(seed=42):
    np.random.seed(seed)
    out = []

    wlv = np.array([240, 245, 250, 255, 260, 265])
    ints = np.random.randn(4, 6)
    a = HSI.Spectra(ints, wlv, ['a' for i in range(len(ints))])
    out.append(a)

    wlv = np.array([244, 249, 254, 259, 264])
    ints = np.random.randn(3, 5)
    b = HSI.Spectra(ints, wlv, ['b' for i in range(len(ints))])
    out.append(b)

    wlv = np.array([246, 251, 256, 260, 266, 271])
    ints = np.random.randn(3, len(wlv))
    c = HSI.Spectra(ints, wlv, ['c' for i in range(len(ints))])
    out.append(c)

    wlv = np.array([250, 255, 260])
    ints = np.random.randn(3, len(wlv))
    d = HSI.Spectra(ints, wlv, ['d' for i in range(len(ints))])
    out.append(d)

    return out