import logging

import numpy as np
import pathlib

from pyperspectral.utils import fold_matrix, unfold_cube
from pyperspectral.constants import RED_NM, GREEN_NM, BLUE_NM


class Hypercube:
    def __init__(self, cube: np.ndarray, wlv: np.ndarray, file_name: str = ""):
        assert np.ndim(cube) == 3
        assert np.ndim(wlv) == 1
        assert cube.shape[-1] == len(wlv)
        self.file_name = file_name
        self.cube = cube
        self.wlv = wlv

    def unfold(self):
        matrix_unfolded = unfold_cube(self.cube)
        return Spectra(matrix_unfolded, self.wlv, self.cube.shape)

    def fake_rgb(self, rgb_vals: tuple = None) -> np.ndarray:
        if rgb_vals is None:
            # Use RGB wavelengths, if they fall in the WLV range
            rgb = np.array([RED_NM, GREEN_NM, BLUE_NM])
            if np.all(self.wlv.min() < rgb) and np.all(rgb < self.wlv.max()):
                rgb_vals = RED_NM, GREEN_NM, BLUE_NM
            else:
                rgb_vals = self.wlv.min(), np.median(self.wlv), self.wlv.max()
        rgb_bin_indeces = [np.abs(self.wlv - wl).argmin() for wl in rgb_vals]  # Find indeces of the closest bins
        rgb_wlv_vals = self.wlv[rgb_bin_indeces]  # Find the corresponding vlaues
        logging.info(f"Using fake RGB vals: {rgb_wlv_vals}")
        fake_rgb: np.ndarray = self.cube[:, :, rgb_bin_indeces]
        return fake_rgb

    def __repr__(self):
        return f"Hypercube(cube={self.cube}, wlv={self.wlv}, file_name={self.file_name}"

    def __str__(self):
        _fname = self.file_name
        if not _fname:
            _fname = "[no file name]"
        return f"Hypercube of dims {self.cube.shape}. WLV: {len(self.wlv)} bins from {self.wlv.min()} to {self.wlv.max()}. File name: {_fname}"


class Spectra:
    def __init__(self, intensities: np.ndarray, wlv: np.ndarray, cube_dims: tuple[int, int, int]):
        assert intensities.shape[-1] == len(wlv)
        self.intensities = intensities
        self.wlv = wlv
        self.cube_dims = cube_dims

    def fold(self, target_shape: tuple = None) -> Hypercube:
        if target_shape is None:
            target_shape = self.cube_dims
        cube = fold_matrix(self.intensities, target_shape)
        return Hypercube(cube, self.wlv)

    @property
    def shape(self):
        return self.intensities.shape

    def __repr__(self):
        return f"Spectra(intensities={self.intensities}, wlv={self.wlv}, cube_dims={self.cube_dims}"

    def __str__(self):
        return f"Spectra object of {self.intensities.shape[0]:,} spectra. WLV: {len(self.wlv)} bins from {self.wlv.min()} to {self.wlv.max()}"
