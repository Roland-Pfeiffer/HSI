import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import load_sample_spectra
import HSI

np.set_printoptions(threshold=np.inf,
                    linewidth=200)

# a_intensities = np.array([[2, 4.2, 5.2, 5, 4, 5, 2.8], [5, 5.2, 5.5, 5.4, 5.3, 5.3, 3.7]])
# a_wlv = np.array([0, 1, 2, 3, 4, 5, 6])
# b_intensities = np.array([[1, 1.1, 1.7, 1.9, 1.8, 1.9, 1.2], [0.1, 0.7, 0.8, 0.6, 0.5, 0.45, 0.31]])
# b_wlv = np.array([-0.9, 0.1, 1.1, 2.1, 3.1, 4.1, 5.1])
# plt.figure('Spectra')
# plt.plot(a_wlv, a_intensities.T, 'bo-', color='blue')
# plt.grid()
# plt.plot(b_wlv, b_intensities.T, 'bo--', color='red')
# plt.xlabel('Wavelength')
# plt.ylabel('Intensity')
# plt.show()

fname_PE = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PE.csv'
data = np.loadtxt(fname_PE, delimiter=';')
PE = HSI.Spectra(data[:, 1], data[:, 0])
# PE.plot()

fname_PLLA = '/media/findux/DATA/HSI_Data/reference_spectra_josef/PLLA.CSV'
data = np.loadtxt(fname_PLLA, delimiter=';')
PLLA = HSI.Spectra(data[:, 1], data[:, 0])
# PLLA.plot()

wlv_a = np.array([3970.88037357, 3972.80892158, 3974.7374696,  3976.66601761, 3978.59456562, 3980.52311363,
                  3982.45166164, 3984.38020965, 3986.30875767, 3988.23730568, 3990.16585369, 3992.0944017,
                  3994.02294971, 3995.95149773, 3997.88004574, 3999.80859375])
wlv_b = np.array([3970.88, 3972.809, 3974.737, 3976.666, 3978.594, 3980.523, 3982.452, 3984.38, 3986.309, 3988.237,
                  3990.166, 3992.094, 3994.023, 3995.951, 3997.88,  3999.809])
# print(len(wlv_a))
# print(len(wlv_b))
# print(np.round(wlv_a, 3))
# print(np.round(wlv_b, 3))
# print(PE.wlv)
# print()
# print(PLLA.wlv)


a = np.array([3700.884, 3702.812, 3704.741, 3706.669, 3708.598, 3710.526, 3712.455, 3714.383, 3716.312, 3718.24,
              3720.169, 3722.098, 3724.026, 3725.955, 3727.883, 3729.812, 3731.74, 3733.669, 3735.597, 3737.526,
              3739.455, 3741.383])
b = np.array([3700.88365191, 3702.81219992, 3704.74074793, 3706.66929595, 3708.59784396, 3710.52639197, 3712.45493998,
              3714.38348799, 3716.31203601, 3718.24058402, 3720.16913203, 3722.09768004, 3724.02622805, 3725.95477606,
              3727.88332408, 3729.81187209, 3731.7404201, 3733.66896811, 3735.59751612, 3737.52606414, 3739.45461215,
              3741.38316016])
# print(len(a))
# print(len(b))
# print(np.round(a, 2))
# print(np.round(b, 2))

start, stop = 208, 220
a_intensities = PLLA.intensities[start:stop]
a_wlv = PLLA.wlv[start:stop]
# print(np.round(a_wlv, 3))

start, stop = 65, 77
b_intensities = PE.intensities[start:stop]
b_wlv = PE.wlv[start:stop]

print(a_intensities)
print(a_wlv)
print(b_intensities)
print(b_wlv)

a_intensities = np.array([0.2184649, 0.2201522, 0.2186561, 0.2158286, 0.2134198, 0.2112872, 0.2141369, 0.2177338, 0.2169663, 0.2109826, 0.2084285, 0.2115282])
a_wlv = np.array([800.3474, 802.2759, 804.2045, 806.1331, 808.0616, 809.9902, 811.9187, 813.8472, 815.7758, 817.7043, 819.6329, 821.5615])
b_intensities = np.array([0.02411663, 0.02425605, 0.02463717, 0.02500274, 0.0251241, 0.02488419, 0.0243349, 0.02371608, 0.02327189, 0.023061, 0.023011, 0.02312733])
b_wlv = np.array([800.34744206, 802.27599007, 804.20453808, 806.1330861, 808.06163411, 809.99018212, 811.91873013, 813.84727814, 815.77582616, 817.70437417, 819.63292218, 821.56147019])

plt.plot(a_wlv, a_intensities, 'bo-', color='red')
plt.plot(b_wlv, b_intensities, 'bo-', color='green')
plt.grid()
plt.show()
