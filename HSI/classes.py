import numpy as np
import pathlib

from HSI.utils import fold_matrix, unfold_cube


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

    def fake_rgb(self, r: float = None, g: float = None, b: float = None):
        """

        :param r: red wavelength
        :param g: green wavelength
        :param b: blue wavelength
        :return:
        """
        wl_min, wl_max = self.wlv.min(), self.wlv.max()

        # TODO: if no rgb values are provided, use min, max and median as bins.

        assert np.all(wl_min <= val <= wl_max for wl in (r, g, b))
        rgb_bins = [self.wlv[np.abs(self.wlv - wl).argmin()] for wl in [r, g, b]]

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


