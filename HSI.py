#!/usr/bin/env python3
from __future__ import annotations  # F. Spectra class w/ type hint in method .add_spectra referring to own parent class
import spectral
import numpy as np
import scipy.stats
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Union  # So multiple types can be specified in function annotations
import warnings
import logging


# ToDo: Think about what to do when using different WLVs for different samples.
#       would dictionaries be too slow?
#       This is actually the case with Josef's reference spectra.

# ToDo: Take into account peak prominence.

# ToDo: normalisation, mean centering (in preprocessing function?)

def load_hsi(fpath: str) -> 'hdr, img, wlv':
    """spectra = load_hsi(fpath)\n\n
    Takes path to a header (.hdr) hsi file and returns header file, hypercube array and wavelength vector (WLV)
    (aka wavenumbers). WLV is retrieved from the centers of bands.
    To load just one pixel's spectrum, use load_pixel().
    :rtype: .hdr, np.array, np.array
    """
    hdr = spectral.open_image(fpath)
    img_cube = hdr.load()
    wlv = np.array(hdr.bands.centers)
    # spct = Spectra(unfold_cube(img_cube), wlv)
    return hdr, img_cube, wlv


def load_pixel(hdr_fpath: str, row, col, material: str = None) -> 'spectrum: Spectra':
    """spectrum, wlv = load_pixel(fpath, row, col)
    Reads the spectrum of given pixel and outputs a spectrum object.
    """
    hdr = spectral.open_image(hdr_fpath)
    spec = hdr.read_pixel(row, col)
    wlv = hdr.bands.centers
    out = Spectra(spec, wlv, list(material))
    return out


def unfold_cube(cube):
    """spectra = unfold_cube(cube)\n\n
    Unfolds a hypercube of the dimensions (x, y, z) into a 2D numpy array
    of the dimensions (x * y, z) containing the spectra for each pixel.
    """
    _cubearray = np.array(cube)
    _x, _y, _spec = _cubearray.shape
    spectra = _cubearray.reshape((_x * _y, _spec))
    return spectra


def align_wlv(wlv_to_align, wlv_to_align_WITH):
    """Aligns WLV a with WLV b.
    Returns the aligned version of WLV a as a numpy array."""
    if min(wlv_to_align) < min(wlv_to_align_WITH):
        print('wlv a begins with lower value(s) than reference wlv b.')
    if max(wlv_to_align) > max(wlv_to_align_WITH):
        print('wlv a overhangs reference wlv b (max(a) > max(b)).')
    return [wlv_to_align_WITH[np.argmin(abs(wlv_to_align_WITH - k))] for k in wlv_to_align]


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
        _mask = np.where(_mask > 0.5, 1, 0)  # = where mask>0.5 return 1, else 0.
        _x, _y = _mask.shape
        self.mask_2D = _mask
        self.full_vector = _mask.reshape((_x * _y))  # Unfold
        self.index_vector = np.where(self.full_vector == 1)[0]  # Locate only "in" values
        self.material = material


def load_and_unfold(hdr_path, mask = None, material = None):
    _hdr, _cube, _path = load_hsi(hdr_path)


def points_within_wlv(points: iter(), wlv:np.ndarray):
    return min(points) >= min(wlv) and max(points) <= max(wlv)


class Spectra:
    """2D np.array containing a set of spectra.
    """
    def __init__(self, intensities: np.array, wlv: np.array, material_column: list = None):
        self.wlv = wlv
        self.intensities = intensities
        self.material_column = material_column

    def random_subsample(self, n=250, seed: int = 42):
        """
        Returns a random subsample of the Spectra object.
        Note that this does not work inplace, but needs to be assigned to a new var.
        """
        assert n > 0
        # Account for n >= numbe of spectra:
        if n >= self.intensities.shape[0]:
            return Spectra(self.intensities, self.wlv)
        # Otherwise, take random sample
        random.seed(seed)
        subset_index = random.choices(range(self.intensities.shape[0]), k=n)
        return Spectra(self.intensities[subset_index], self.wlv, self.material_column)

    def add_spectra(self, spectra: Spectra):
        if not np.alltrue(self.wlv == spectra.wlv):
            raise Exception("Wavelength vectors are not the same.\nSpectra not merged.")
        else:
            self.intensities = np.vstack((self.intensities, spectra.intensities))
            # Update material column
            [self.material_column.append(material) for material in spectra.material_column]

    def export_to_npz(self, savename):
        np.savez(savename, self.intensities, self.wlv)

    def plot(self):
        x = self.wlv
        y = self.intensities
        plt.plot(x, y.T)
        plt.show()

    def smoothen(self, window_size, polynomial: int, derivative: int = 0):
        self.intensities = savgol_filter(self.intensities,window_size, polynomial, derivative)

    def verify(self):
        """Returns True if wlv len fits the spectral dimension of the intensities matrix"""
        if len(self.wlv) == self.intensities.shape[1]:
            return True
        else:
            return False


class MulticlassMask:
    def __init__(self, masks: tuple, materials: tuple):
        """Takes a tuple of binary masks and a tuple of materials and turns them into a multiclass masking vector."""
        assert len(masks) == len(materials)
        self.masks = masks
        self.materials = materials
    # ToDo: Combine binary masks, material = mask fname.
    pass


def mask_spectra(spectra: Spectra, mask: BinaryMask):
    pass


def find_peaks(spectrum: np.array, wlv: np.array):
    pass


class Descriptor:
    """General descriptor class. Makes sure they all have a .material attribute.
    All descriptors (triangles, etc.) inherit from this.
    """
    def __init__(self, mat=None):
        self.material = mat
    # ToDo: add start/stop and offer calculating descriptor integral.

class TriangleDescriptor(Descriptor):
    """Takes a start wavelength, peak wavelengths and stop wavelength, and material  as input.
    """
    def __init__(self, wl_start: Union[int, float], wl_peak: Union[int, float], wl_stop: Union[int, float],
                 material: str = 'Material'):
        # Input validation
        if not wl_start < wl_peak < wl_stop:
            raise ValueError('Invalid start, peak and stop input. Need to be numerical and start < peak < stop.')

        super().__init__(material)
        self.start_wl = wl_start
        self.peak_wl = wl_peak
        self.stop_wl = wl_stop
        # Wavelength attributes for start, peak and stop
        # Initiate index attributes
        start_bin_index, peak_bin_index, stop_bin_index = None, None, None

    def compare_to_spectrum(self, spectrum, wlv: np.array, region_divisor: int = 2):
        """(avg_pearson_r, avg_r_multiplied_by_rel_peak_height) = TriangleDescriptor.compare_to_spectrum(spectrum, wlv, region_divisor)
        Takes a Spectrum as input and then compares how well it is matched by the descriptors.
        Returns average pearson correlation as well as avg. correl. muliplied by relative peak height.
        region_divisor: number of bins before and after peak will be divided by this. Is used as avg. region width.
        ToDo:   Perhaps use a Spectra class as input, making the second wlv parameter obsolete. However, now it's more
                accessible.
        ToDo:   Change region divisor to region width or something more relatable
        """
        # Account for values outside of wlv range:
        assert min(wlv) < self.peak_wl < max(wlv)
        if self.start_wl <= min(wlv):
            self.start_wl = min(wlv)
        if self.stop_wl >= max(wlv):
            self.stop_wl = max(wlv)

        # Get bin indices
        start_bin_index = np.argmin(np.abs(wlv - self.start_wl))
        peak_bin_index = np.argmin(np.abs(wlv - self.peak_wl))
        stop_bin_index = np.argmin(np.abs(wlv - self.stop_wl))
        logging.debug('Index: (Start|Peak|Stop): ({0}|{1}|{2})'.format(start_bin_index, peak_bin_index, stop_bin_index))

        # Create linspaces (+1 because peak and stop bin are included)
        before_peak = int(peak_bin_index - start_bin_index) + 1
        after_peak = int(stop_bin_index - peak_bin_index) + 1

        asc_linspace = np.linspace(0, 1, before_peak)
        desc_linspace = np.linspace(1, 0, after_peak)

        logging.debug('Before peak: {}'.format(before_peak))
        logging.debug('First linspace: {}'.format(asc_linspace))
        logging.debug('After peak: {}'.format(after_peak))
        logging.debug('Second linspace: {}'.format(desc_linspace))

        # Get peak height (avg. of peak region - avg. of start/stop region (depending which is lower))
        # Make sure the descriptor is wide enough:
        if before_peak // region_divisor < 1:
            start_avg = spectrum[start_bin_index]
        else:
            start_avg = np.mean(spectrum[start_bin_index - (before_peak // region_divisor):
                                         start_bin_index + (before_peak // region_divisor)])
        if before_peak // region_divisor < 1 or after_peak // region_divisor < 1:
            peak_avg = spectrum[peak_bin_index]
        else:
            peak_avg = np.mean(spectrum[peak_bin_index - (before_peak // region_divisor):
                                        peak_bin_index + (after_peak // region_divisor)])
        if after_peak // region_divisor < 1:
            stop_avg = spectrum[stop_bin_index]
        else:
            stop_avg = np.mean(spectrum[stop_bin_index - (after_peak // region_divisor):
                                        stop_bin_index + (after_peak // region_divisor)])

        low = min([start_avg, stop_avg])
        peak_height = peak_avg - low
        rel_peak_height = peak_height / (max(spectrum) - min(spectrum))
        logging.debug('Peak height: {}'.format(peak_height))
        logging.debug('Relative peak height: {}'.format(rel_peak_height))


        # Calculate Pearson's Correlation Coefficient (r):
        pre_peak_intensities = spectrum[start_bin_index:peak_bin_index + 1]
        post_peak_intensities = spectrum[peak_bin_index:stop_bin_index + 1]

        logging.debug('Pre-peak intensities: {}'.format(pre_peak_intensities))
        logging.debug('Post-peak intensities: {}'.format(post_peak_intensities))

        pearsons_r_asc = scipy.stats.pearsonr(pre_peak_intensities, asc_linspace)
        pearsons_r_desc = scipy.stats.pearsonr(post_peak_intensities, desc_linspace)
        logging.debug('Peasons r (ascending|descending): ({0}|{1})'.format(pearsons_r_asc, pearsons_r_desc))

        pearsons_r_avg = (pearsons_r_asc[0] + pearsons_r_desc[0]) / 2
        # "Deactivate" everything below 0.5:
        if pearsons_r_avg >= 0.5:
            out = pearsons_r_avg * rel_peak_height
        else:
            out = 0
        return out

    def plot(self):
        print('Still to be done...')

    def show(self):
        return 'TriangleDescriptor:\t(Start: {0} | Peak: {1} | Stop: {2}. Material: {3})'.\
            format(self.start_wl, self.peak_wl, self.stop_wl, self.material)

    # Output when print() is run on the descriptor:
    def __str__(self):
        return 'TriangleDescriptor:\t(Start: {0} | Peak: {1} | Stop: {2}. Material: {3})'.\
            format(self.start_wl, self.peak_wl, self.stop_wl, self.material)


class DescriptorSet:
    def __init__(self, descriptor):
        self.descriptors = [descriptor]
        self.material = descriptor.material

    def add_descriptor(self, descriptor):
        if not self.material.lower() == descriptor.material.lower():
            raise IOError('Material are not the same.')
        self.descriptors.append(descriptor)

    def show_materials(self):
        for _D in self.descriptors:
            print(_D.material)

    def __str__(self):
        _out = ''
        for i in range(len(self.descriptors)):
            _out += str(self.descriptors[i]) + '\n'  # str(object) returns what the object gives when print()-ed
        return _out

    def correlate(self, spectra: np.array, wlv: np.array, region_divisor = 2):
        """Note that the correlation matrix includes values that have been """
        descriptor_count = len(self.descriptors)
        spectra_count = len(spectra)
        corr_mat = np.zeros((spectra_count, descriptor_count))
        for spec in range(spectra_count):
            for desc in range(descriptor_count):
                corr_mat[spec, desc] = self.descriptors[desc].compare_to_spectrum(spectra[spec], wlv, region_divisor)
        names = [self.descriptors[i].material + str(i) for i in range(descriptor_count)]
        data_out = pd.DataFrame(corr_mat)
        rename_dict = dict(zip(range(descriptor_count), names))
        data_out.rename(columns=rename_dict, inplace=True)
        return data_out
