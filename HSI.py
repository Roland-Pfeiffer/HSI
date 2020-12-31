#!/usr/bin/env python3
import spectral
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from typing import Union  # So multiple types can be specified in function annotations


# ToDo: Think about what to do when using different WLVs for different samples.
#       would dictionaries be too slow?
#       This is actually the case with Josef's reference spectra.


def load_hsi(fpath: str) -> 'hdr, img, wlv':
    """hdr, img, wlv = hsi_import(fpath)\n\n
    Takes path to a header (.hdr) hsi file and returns
    header file, hypercube array and wavelength vector (WLV)
    (aka wavenumbers).
    WLV is retrieved from the centers of bands.
    :rtype: .hdr, np.array, np.array
    """
    hdr = spectral.open_image(fpath)
    img_cube = hdr.load()
    wlv = np.array(hdr.bands.centers)
    return hdr, img_cube, wlv

# def preprocessing(spectra: Spectra):
#
#     pass
#     # ToDo: normalisation, mean centering


def align_with_bins(wlv_bins: np.array, wl: np.array):
    """Outputs the nearest bin center value for wavelengths that
    are not exactly matching with the bin centers.\n\n
    Adapted from: https://stackoverflow.com/a/26026189
    """
    idx = np.searchsorted(wlv_bins, wl, side="left")
    if idx > 0 and (idx == len(wlv_bins)
                    or math.fabs(wl - wlv_bins[idx - 1]) < math.fabs(wl - wlv_bins[idx])):
        return wlv_bins[idx - 1]
    else:
        return wlv_bins[idx]



def find_peaks(spectrum: np.array, wlv: np.array):
    pass


class Spectra:
    """2D np.array containing a set of spectra.
    """
    def __init__(self, intensities: np.array, wlv: np.array, material_column: list = None):
        self.wlv = wlv
        self.intensities = intensities
        self.material_column = material_column

    def random_subsample(self, n=250, seed=42):
        subset_index = random.choices(range(self.intensities.shape[0]), k=n) # ToDo: Incorporate seed
        return Spectra(self.intensities[subset_index], self.wlv, self.material_column)

    def export_npz(self, savename):
        np.savez(savename, self.intensities, self.wlv)

    def plot(self):
        x = self.wlv
        y = self.intensities
        plt.plot(x, y.T)
        plt.show()


def unfold_cube(cube):  # Rename to unfold cube?
    """spectra = spectra_from_cube(cube)\n\n
    Unfolds a hypercube of the dimensions (x, y, z) into a 2D array
    of the dimensions (x * y, z) containing the spectra for each pixel.
    """
    _cubearray = np.array(cube)
    _x, _y, _spec = _cubearray.shape
    spectra = _cubearray.reshape((_x * _y, _spec))
    return spectra


def random_spectrum(spectra):  # Can be replaced by "Spectra.random_subsample(n=1)
    pass


class BinaryMask:
    """
    Binary mask object with the following attributes:
    .mask_2D        binary 2D array
    .full_vector    unfolded mask (with 0 for out-, 1 for in-values)
    .index_vector   unfolded mask vector containing indices for in-values
    .material       Material string
    """
    def __init__(self, img_path: str, material: str):
        _mask = plt.imread(img_path)[:, :, 0]
        # Crank everything above 50% intensity up to 100%:
        _mask = np.where(_mask > 0.5, 1, 0)
        _x, _y = _mask.shape
        self.mask_2D = _mask
        self.full_vector = _mask.reshape((_x * _y))  # Unfold
        self.index_vector = np.where(self.full_vector == 1)[0]  # Locate only "in" values
        self.material = material

class MulticlassMask:
    # ToDo: Combine binary masks, material = mask fname.
    pass


def mask_spectra(spectra: Spectra, mask: BinaryMask):
    pass


class TriangleDescriptor:
    """ToDo: Break this up.

    Takes a start wavelength, peak wavelengths and stop wavelength, as well
    as a WavelengthVector object as input.
    """
    def __init__(self, material_name: str,
                 wl_start: Union[int, float], wl_peak: Union[int, float], wl_stop: Union[int, float],
                 wlv: np.array):
        self.material_name = material_name
        # Wavelength values validation
        if not wl_start < wl_peak < wl_stop:
            raise ValueError('Invalid wavelengths input.\n'
                             'Are they float or int?\n'
                             'Are they in order START, PEAK, STOP?')
        # Wavelength attributes for start, peak and stop
        self.start_wl = align_with_bins(wl_start)
        self.peak_wl = align_with_bins(wl_peak)
        self.stop_wl = align_with_bins(wl_stop)
        # Initiate bin centers
        self.start_bin = None
        self.peak_bin = None
        self.stop_bin = None
        # Initiate index attributes
        self.start_bin_index = None
        self.peak_bin_index = None
        self.stop_bin_index = None
        # Initiate linspace
        self.asc_linspace = None
        self.desc_linspace = None

    def compare_to_spectrum(self, spectrum, wlv): # ToDo: Maybe separate this into its own function
        """
        Takes a Spectrum as input and then compares how well it is matched by the descriptors.
        ToDo: The actual comparison.
        :param spectrum:
        :return:
        """
        # CHECK IF THE PEAK LOCATIONS ARE OKAY AND ADJUST FOR OUT-OF-RANGE VALUES.
        if self.peak_wl < min(wlv) or self.peak_wl > max(wlv):
            raise ValueError('Descriptor peak falls outside of spectrum\'s wavelength vector.')
        # If all values are within spectrum WLV range:
        if min(wlv) <= self.start_wl < self.peak_wl < self.stop_wl <= max(wlv):
            self.start_bin = align_with_bins(wlv, self.start_wl)
            self.peak_bin = align_with_bins(wlv, self.peak_wl)
            self.stop_bin = align_with_bins(wlv, self.stop_wl)
        # Out-of-range values:
        elif self.start_wl < min(wlv) < self.peak_wl < self.stop_wl <= max(wlv):
            self.start_bin = min(wlv)
            print('Starting wavelength coincides with spectrum\'s WLV range limit or lies beyond.\n'
                  'Set to minimum WLV value.')
            self.peak_bin = align_with_bins(wlv, self.peak_wl)
            self.stop_bin = align_with_bins(wlv, self.stop_wl)
        elif min(wlv.wavelengths) <= self.start_wl < self.peak_wl < max(wlv.wavelengths) <= self.stop_wl:
            self.start_bin = align_with_bins(wlv, self.start_wl)
            self.peak_bin = align_with_bins(wlv, self.peak_wl)
            self.stop_bin = max(wlv)
            print('Starting wavelength coincides with spectrum\'s WLV range limit or lies beyond.\n'
                  'Set to maximum WLV value.')
        # Both values out of range (Unlikely. Consider raising a value error.)
        elif self.start_wl < min(wlv) < max(wlv) < self.stop_wl:
            self.start_bin = min(wlv)
            self.peak_bin = align_with_bins(wlv, self.peak_wl)
            self.stop_bin = max(wlv)
            print('Starting and stopping wavelengths coincide with spectrum\'s WLV range limit or lie beyond.\n'
                  'Set to minimum and maximum WLV value, respectively.')
        else:
            raise ValueError('ERROR: Setting start, peak and stop values failed.')

        # Get bin indices (for spectrum wlv)
        self.start_bin_index = np.where(wlv.wavelengths == self.start_bin)[0]
        self.peak_bin_index = np.where(wlv.wavelengths == self.peak_bin)[0]
        self.stop_bin_index = np.where(wlv.wavelengths == self.stop_bin)[0]

        # ToDo: create in-index-mask to select descriptor range for cutting spectrum to descriptor range
        #       call it descriptor_range_indices
        # Or:   use "spectra[:, start_bin_i:stop_bin_i + 1]

        before_peak = int(self.peak_bin_index - self.start_bin_index)
        after_peak = int(self.stop_bin_index - self.peak_bin_index)

        self.asc_linspace = np.linspace(0, 1, before_peak)
        self.desc_linspace = np.linspace(1, 0, after_peak)
        print(self.asc_linspace)
        print(self.desc_linspace)

        # ToDo: Create linspace
        # ToDo: Run pearson:

    # Output when print() is run on the descriptor:
    def __str__(self):
        return 'HSI.TriangleDescriptor: {0} (start, peak, stop)'\
            .format((self.start_wl, self.peak_wl, self.stop_wl))


def pearson_corr_coeff(descriptors, samples):
    # read wlv only once
    pass


class reference_spectra:
    def __init__(self):
        pass