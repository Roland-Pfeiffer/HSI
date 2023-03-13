#!/usr/bin/env python3
import pandas as pd
import math


def fov(foc_len, sensor_x_y):
    c = sensor_x_y[0] / 2
    print(f'c: {c}')
    a = math.sqrt((c ** 2) + (foc_len ** 2))
    print(f'a: {a}')
    f_ov = math.asin(c / a)  # returns the result in radians
    return 2 * math.degrees(f_ov)


sensor_dims = (9.6, 7.2)
width = 640
F = 2.8

print(fov(35, sensor_dims))

# focal_lengths = [16, 25, 35, 50]
# altitude_m = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
#
# data = pd.DataFrame(altitude_m, columns=['Altitude_m'])
#
#
# print(data)
