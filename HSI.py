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
#       This is actually the case with Josef's reference spectra.
#       would dictionaries be too slow?
# ToDo: Take into account peak prominence.
# ToDo: normalisation, mean centering (in preprocessing function?)
# ToDo: Allow masking when creating a spectra object
# ToDo: align_wlv: Also produce an error estimate between the WLVs
# ToDo: align_wlv: get it to work for different resolutions (scipy interpolate?)
# ToDo: load_and_unfold(): finish this and add mask functionality
# ToDo: load_area()
# ToDo: mask_spectra()
# ToDo: find_peaks()
# ToDo: Spectra.add_spectra(): mention skipped spectra, don't stop the script w/ assert (as is the case now)
# ToDo: Spectra.plot(): add a material legend.
# ToDo: Spectra.plot(): (in the same vein) add a group-my-material option
# ToDo: TriangleDescriptor.compare_to_spectrum(): - Perhaps use a Spectra class as input, making the second wlv
#                                                   parameter obsolete. However, now it's more accessible.
#                                                 - turn "get intensities" into a function.
#                                                 - fix line: return pearsons_r * region_span_rel
# ToDo: DescriptorSet.correlate(): replace index or enumerate()

def load_hsi(fpath: str) -> 'hdr, cube, wlv':
    """
    spectra = load_hsi(fpath)
    Takes path to a header (.hdr) hsi file and returns header file, hypercube array and wavelength vector (WLV) with the
    wavelengths in nm (not wavenumbers in cm-1!). WLV is retrieved from the centers of bands.
    :param fpath: path to .hdr file of the hyperspectral image
    :return: (hdr, cube, wlv)
    """
    """
    :param fpath: path to .hdr file of the hyperspectral image
    :return: tuple of hdr, cube and wlv

    
    To load just one pixel's spectrum, use load_pixel().
    :rtype: .hdr, np.array, np.array
    """
    hdr = spectral.open_image(fpath)
    cube = hdr.load()
    wlv = np.array(hdr.bands.centers)
    # spct = Spectra(unfold_cube(img_cube), wlv)
    return hdr, cube, wlv


def load_pixel(hdr_fpath: str, row, col, material: str = None) -> 'spectrum: Spectra':
    """
    @param hdr_fpath:
    @type hdr_fpath:
    @param row:
    @type row:
    @param col:
    @type col:
    @param material:
    @type material:
    @return:
    @rtype:
    """
    hdr = spectral.open_image(hdr_fpath)
    spec = hdr.read_pixel(row, col)
    wlv = np.array(hdr.bands.centers)
    out = Spectra(spec, wlv, [material])
    return out


def load_area(corner_tl: tuple, corner_br: tuple, material: str=None):
    """Loads a rectangular area out of a hyperspectral datacube.
    :param corner_tl: Tuple containing of (row, column) of the TL corner
    :param corner_br: Tuple containing of (row, column) of the BR corner"""
    pass
    # ToDo


def load_and_unfold(hdr_path, mask=None):
    _hdr, _cube, _path = load_hsi(hdr_path)
    # ToDo: finish this and add mask functionality


def unfold_cube(cube):
    """spectra = unfold_cube(cube)\n\n
    Unfolds a hypercube of the dimensions (x, y, z) into a 2D numpy array
    of the dimensions (x * y, z) containing the spectra for each pixel.
    """
    _cubearray = np.array(cube)
    _x, _y, _spec = _cubearray.shape
    spectra = _cubearray.reshape((_x * _y, _spec))
    return spectra


def mask_spectra(spectra: Spectra, mask: BinaryMask):
    pass  # ToDo: mask spectra


def find_peaks(spectrum: np.array, wlv: np.array):
    pass


def align_wlv(wlv_a, reference_wlv):
    """Aligns wavelength vector (WLV) a with reference WLV.
    Returns the aligned version of wavelength vector a a as a numpy array."""
    # ToDo: Also produce an error estimate between the WLVs
    # ToDo: Get it to work for WLVs of different resolutions
    # Left overhang
    if min(wlv_a) < min(reference_wlv):
        print('wlv 1 begins with lower value(s) than reference wlv {1}.'.format(wlv_a, reference_wlv))
        print('NOTE: min(wlv_to_align) < min(wlv_to_align_WITH),\n'
              'First wlv overhangs reference wlv on the lower end.\n'
              f'{wlv_a}\n{reference_wlv}\n'
              'Overhanging areas will be filled with min() of reference vector.')
    # Right overhang
    if max(wlv_a) > max(reference_wlv):
        print('NOTE: max(wlv_to_align) > max(wlv_to_align_WITH),\n'
              'First wlv overhangs reference wlv on the upper end.\n'
              f'{wlv_a}\n{reference_wlv}\n'
              'Overhanging areas will be filled with max() of reference vector.')
    aligned_wlv = [reference_wlv[np.argmin(abs(reference_wlv - k))] for k in wlv_a]
    return aligned_wlv


def wavelen_to_wavenum(wl_nm: Union[float, int]):
    """
    Takes wavelength (in nm) and returns the wavenumber (per cm⁻¹)
    :param wl_nm: Wavelength (in nm)
    :return: wavemunber (cm ⁻¹)
    """
    return 10_000_000 / wl_nm  # 10 mio nm in one cm


def wavenum_to_wavelen(wavenum_cm1: Union[float, int]):
    """Takes wavenumber (per cm⁻¹) and returns wavelength (in nm)"""
    return 10_000_000 / wavenum_cm1  # 10 mio nm in one cm


def points_within_wlv(points: iter(), wlv:np.ndarray):
    """Checks if points (an iterable) all lie within a WLV.
    Returns True or False"""
    return min(points) >= min(wlv) and max(points) <= max(wlv)


def correlation(a: np.array, b: np.array):
    """Returns the Pearson correlation (R) of a and b"""
    assert len(a) == len(b), 'Vectors for Pearson correl. are of unequal length.'
    pearson_r = scipy.stats.pearsonr(a, b)
    return pearson_r


class BinaryMask:
    """
    Binary mask object.
    Attributes:
        .mask_2D        binary 2D array
        .full_vector    unfolded mask (with 0 for out-, 1 for in-values)
        .in_indices     unfolded mask vector containing indices for in-values
        .material       Material string
    """
    def __init__(self, img_path: str, material: str):
        _mask = plt.imread(img_path)[:, :, 0]
        # Crank everything above 50% intensity up to 100%:
        _mask = np.where(_mask > 0.5, 1, 0)  # = where mask>0.5 return 1, else 0.
        _x, _y = _mask.shape
        self.mask_2D = _mask
        self.full_vector = _mask.reshape((_x * _y))  # Unfold
        self.in_indices = np.where(self.full_vector == 1)[0]  # Locate only "in" values
        self.material = material


class Spectra:
    """
    Class containing a 2D np.array of the intensities (intensities), a wavelength vector (wlv) and a list of the
    materials of each intensity entry (can be None).
    """
    def __init__(self, intensities: np.array, wlv: np.array, material: list = None):
        """
        @param intensities: np.array. Dim 0: pixel, Dim 1: wavelengths. 1D will be forced into 2D.
        @param wlv: 1D np.array
        @param material: Material string (or None).
        """
        self.wlv = wlv
        self.intensities = np.atleast_2d(intensities)  # Needs to be a 2D array, even when just 1 pixel.
        logging.info(f'WLV len: {len(self.wlv)}')
        logging.info(f'Ints. shape: {self.intensities.shape}')
        assert len(self.wlv) == self.intensities.shape[1], ValueError('WLV length does not match intensities.')

        # If the material column is not a list, turn it into one.
        if material is None:
            self.material = ['No_material' for i in range(self.intensities.shape[0])]
        elif len(material) == 1:
            self.material = material * self.intensities.shape[0]
        elif len(material) == self.intensities.shape[0]:
            self.material = material
        else:
            raise ValueError('Ambiguous material column (len neither 1 nor similar to pixel count).')

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
        return Spectra(self.intensities[subset_index], self.wlv, self.material)

    def add_spectra(self, new_spectra: Spectra):
        if not np.alltrue(self.wlv == new_spectra.wlv):
            print('ERROR: WLVs not identical.Spectra not merged.')
        else:
            # ToDo: add a function to align wlvs
            # ToDo: Maybe just note that it was skipped so it doesn't break the code
            # 'Glue' the new spectra under the existing one
            self.intensities = np.vstack((self.intensities, new_spectra.intensities))
            # Update (i.e. append) material column
            if new_spectra.material is not None:
                self.material += new_spectra.material
            else:
                [self.material.append('No_material') for i in range(new_spectra.intensities.shape[0])]

    def export_to_npz(self, savename):
        np.savez(savename, self.intensities, self.wlv)

    def plot(self):
        """
        Plots the spectra, colored by material.
        """
        def legend_without_duplicate_labels(ax):
            """From: https://stackoverflow.com/a/56253636"""
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))

        fig, ax = plt.subplots()
        for i, material in enumerate(np.unique(self.material)):
            i_in = np.argwhere(np.array(self.material) == material).flatten()
            ax.plot(self.wlv, self.intensities[i_in].T,
                    label=material,
                    color=next(ax._get_lines.prop_cycler)['color'])
        legend_without_duplicate_labels(ax)
        ax.grid()
        plt.show()

    def smoothen_savgol(self, window_size, polynomial: int):
        """Returnes a Spectra object with smoothed intensity array"""
        ints_smoothed = savgol_filter(self.intensities, window_size, polynomial, 0)
        return Spectra(ints_smoothed, self.wlv, self.material)

    def derivative(self):
        """Returns an array containing the gradients of the intensities. Does not replace them inplace.
        Does not create a new spectra object, just returns a np.array"""
        assert self.intensities.ndim == 2
        return np.gradient(self.intensities, axis=1)

    def return_intensity_gradient(self):
        """Returns a Spectra object where the 'intensities' array actually contains the (1st) derivatives."""
        grads = np.gradient(self.intensities, axis=1)
        return Spectra(grads, self.wlv, self.material)

    def select_by_material(self, material: str):
        _i_in = np.where(self.material == material)
        logging.info(f'IN indices: {_i_in}')
        return _i_in

    def verify_bin_counts(self):
        """Returns True if wlv len fits the spectral dimension of the intensities matrix"""
        if len(self.wlv) == self.intensities.shape[1]:
            return True
        else:
            return False

    def return_as_df(self, intensities_as_list=True):
        """
        Returns a pd dataframe that contains the intensities as rows, wlv as column names, while the last column
        contains the material.
        @return:
        """
        if intensities_as_list:
            _df = pd.DataFrame({'Intensities': self.intensities.tolist(),
                                'Material': self.material})
        else:
            _df = pd.DataFrame(self.intensities, columns=self.wlv)
            _df['Material'] = self.material
        return _df

    def fake_rgb(self):
        pass


class TriangleDescriptor():
    """Takes a start wavelength, peak wavelengths and stop wavelength, and material  as input.
    """
    def __init__(self, wl_start: Union[int, float], wl_peak: Union[int, float], wl_stop: Union[int, float],
                 material: str = 'No material specified'):
        assert (wl_start < wl_peak < wl_stop), 'Points not numerical or not in correct order (start, peak, stop).'
        self.material = material
        self.start_wl = wl_start
        self.peak_wl = wl_peak
        self.stop_wl = wl_stop
        self.start_i = None
        self.peak_i = None
        self.stop_i = None

    def compare_to_spectrum(self, spectrum, wlv: np.array, multiply_w_region_span=False, prob_threshold=0.5):
        """
        Takes a Spectrum as input and then compares how well it is matched by the descriptors.
        Returns average pearson correlation as well as avg. correl. BOTH muliplied by relative peak height.
        region_divisor: number of bins before and after peak will be divided by this. Is used as avg. region width.
        ToDo:   Perhaps use a Spectra class as input, making the second wlv parameter obsolete. However, now it's more
                accessible.
        """
        # Account for values outside of wlv range.
        # Make sure the peak is BETWEEN the end values
        assert min(wlv) < self.peak_wl < max(wlv), 'Peak wavelength falls outside WLV.'
        # if start is below of wlv min, set wlv min as start_wl
        if self.start_wl <= min(wlv):
            self.start_wl = min(wlv)
        # If stop is above wlv max set wlv max as stop_wl
        if self.stop_wl >= max(wlv):
            self.stop_wl = max(wlv)

        # Get indices for start, peak and stop
        self.start_i = np.argmin(np.abs(wlv - self.start_wl))
        self.peak_i = np.argmin(np.abs(wlv - self.peak_wl))
        self.stop_i = np.argmin(np.abs(wlv - self.stop_wl))
        logging.info(f'Start i: {self.start_i} | Peak i: {self.peak_i} | Stop i: {self.stop_i}')

        # Make sure the points are distinct. (as when e.g. the WLV res. is too low)
        # This at the same time checks that they are in the correct order
        assert (self.start_i < self.peak_i < self.stop_i), 'Points not distinct on WLV or not in correct order (start, peak, stop).'
        # ToDo: make this a non-exit warning

        # Create descriptor vector:
        before_peak_len = int(self.peak_i - self.start_i)
        after_peak_len = int(self.stop_i - self.peak_i)
        last_value_before_peak = 1 - (1 / before_peak_len)
        asc_linspace = np.linspace(0, last_value_before_peak, before_peak_len)
        desc_linspace = np.linspace(1, 0, after_peak_len + 1)  # +1 because stop is included
        triangle_array = np.hstack([asc_linspace, desc_linspace])
        logging.info(f'Triangle array: {triangle_array}')

        # Get intensities
        # ToDo: Turn this into functions
        intensity_start = spectrum[self.start_i]
        intensity_peak = spectrum[self.peak_i]
        intensity_stop = spectrum[self.stop_i]
        logging.info(f'Start: {intensity_start:.4f} [{self.start_i}] | '
                      f'Peak: {intensity_peak:.4f} [{self.peak_i}] | '
                      f'Stop: {intensity_stop:.4f} [{self.stop_i}]')
        # Get relative range of the area (better than peak height, since peak height requires... a peak!
        region_span_abs = np.max(spectrum[self.start_i:self.stop_i + 1]) - np.min(spectrum[self.start_i:self.stop_i + 1])
        region_span_rel = region_span_abs / (np.max(spectrum) - np.min(spectrum) )
        # Calculate Pearson's Correlation Coefficient (r):
        # Extract region of interest from spectrum and compare to the triangle
        spec_roi = spectrum[self.start_i:self.stop_i + 1]
        logging.info(f'Spectrum ROI: {spec_roi}')
        pearsons_r, pearson_p = scipy.stats.pearsonr(triangle_array, spec_roi)

        # "Deactivate" everything below prob threshold and return:
        if abs(pearsons_r) < prob_threshold:
            return 0
        elif multiply_w_region_span:
            return pearsons_r * region_span_rel  # ToDo: FIX
        else:
            return pearsons_r

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
        """Note that the correlation matrix includes values that have been .... [and I never finished the sentence.
        ..... cut off or merged if they were outside of the WLV range?]"""
        descriptor_count = len(self.descriptors)

        # Avoid dimension confusions: Force spectra to be 2D array (using ndmin=2):
        spectra = np.array(spectra, ndmin=2)
        spectra_count = spectra.shape[0]
        logging.info(f'Spectra count: {spectra_count}')
        corr_mat = np.zeros((spectra_count, descriptor_count))
        for spec_i in range(spectra_count):  # ToDo: replace index or enumerate()
            for desc_i in range(descriptor_count):  # ToDo: repl. index or enumerate()
                logging.info(f'Analysing spectrum {spec_i}: descriptor {desc_i}')
                corr_mat[spec_i, desc_i] = self.descriptors[desc_i].compare_to_spectrum(spectra[spec_i, :], wlv)

        correlation_matrix = pd.DataFrame(corr_mat)
        # Create understandable names for the descriptors and assign them to the dataframe
        names = [self.descriptors[i].material + '_desc_' + str(i) for i in range(descriptor_count)]
        rename_dict = dict(zip(range(descriptor_count), names))
        correlation_matrix.rename(columns=rename_dict, inplace=True)
        return correlation_matrix
