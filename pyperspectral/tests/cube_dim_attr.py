import pathlib

from pyperspectral.load import load_hypercube

fpath = "/media/findux/DATA/spectral_data/Martin/2019-05-23_008/results/REFLECTANCE_2019-05-23_008.hdr"
cube = load_hypercube(fpath_hdr=fpath)
spectra = cube.unfold()
print(spectra)
print(cube)