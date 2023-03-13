#!/usr/bin/env python3

import logging
import pathlib

import numpy as np
import spectral

from HSI.classes import Hypercube


def load_hsi(fpath_hdr: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a .hdr file and returns the data cube and a wlv
    :param fpath_hdr:
    :return:
    """
    logging.info(f"Loading file {fpath_hdr}")
    hdr = spectral.open_image(fpath_hdr)
    cube = hdr.load()
    wlv = np.array(hdr.bands.centers)
    logging.info(f"Cube dims: {cube.shape}")
    logging.info(f"WLV dims: {wlv.shape}")
    return cube, wlv


def load_hypercube(fpath_hdr: str) -> Hypercube:
    fname = pathlib.Path(fpath_hdr).name
    cube, wlv = load_hsi(fpath_hdr=fpath_hdr)
    cube = Hypercube(cube=cube, wlv=wlv, file_name=fname)
    return cube


def load_pixel(fpath_hdr: str, pixel_xy: tuple):
    """

    @param fpath_hdr:
    @param pixel_xy:
    @return:
    """
    assert len(pixel_xy) == 2
    col, row = pixel_xy
    hdr = spectral.open_image(file=fpath_hdr)
    px = hdr.read_pixel(row=row, col=col)
