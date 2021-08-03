#!/usr/bin/env python3

import HSI

fpath = '/media/findux/DATA/spectral_data(incomplete)/2019-07-16_003/capture/2019-07-16_003.hdr'
hdr, cube, wlv = HSI.load_hsi(fpath)

print(cube.shape)

HSI.wavelen_to_wavenum()