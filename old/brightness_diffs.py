#!/usr/bin/env python3

import cv2
import spectral
import logging
import pyperspectral

logging.basicConfig(level=logging.DEBUG, format='[ %(levelname)s ] - %(message)s')
#logging.disable()

fpath = input('Enter path to .hdr file.\n >> ')
if fpath.startswith('file://'): fpath = fpath.split('file://')[1]


header, img, wlv = pyperspectral.HSI_import(fpath)

img_rgb = spectral.get_rgb(header)

# ToDo: Everything