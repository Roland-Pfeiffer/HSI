#!/usr/bin/env python3
"""Create test data for alignment"""
import numpy as np
import pyperspectral

def create_alignment_test_data(seed=42):
    np.random.seed(seed)
    out = []

    wlv = np.array([240, 245, 250, 255, 260, 265])
    ints = np.random.randn(4, 6)
    a = pyperspectral.Spectra(ints, wlv, ['a' for i in range(len(ints))])
    out.append(a)

    wlv = np.array([244, 249, 254, 259, 264])
    ints = np.random.randn(3, 5)
    b = pyperspectral.Spectra(ints, wlv, ['b' for i in range(len(ints))])
    out.append(b)

    wlv = np.array([246, 251, 256, 260, 266, 271])
    ints = np.random.randn(3, len(wlv))
    c = pyperspectral.Spectra(ints, wlv, ['c' for i in range(len(ints))])
    out.append(c)

    wlv = np.array([250, 255, 260])
    ints = np.random.randn(3, len(wlv))
    d = pyperspectral.Spectra(ints, wlv, ['d' for i in range(len(ints))])
    out.append(d)

    return out


def load_sample_data():
    pass

if __name__ == '__main__':
    import pyperspectral

    fpaths = ['/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_003/',
              '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_005/',
              '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_007/',
              '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_011/',
              '/media/findux/DATA/spectral_data/SPECIM_field_data/2019-07-16_HS_images_laid_out/2019-07-16_012/']
    # for path in fpaths:
    # file = path.split('/')[-2]
    # mask_path = path + file + '_LITTER.tif'
