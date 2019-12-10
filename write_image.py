import numpy as np
import rasterio
from  rasterio.profiles import DefaultGTiffProfile

BANDS = 3
WIDTH = 17
HEIGHT = 11

array = np.arange(0, BANDS * WIDTH * HEIGHT, dtype='uint16')
array.shape = (BANDS, WIDTH, HEIGHT)

with rasterio.Env():
    profile = DefaultGTiffProfile(
        width=WIDTH,
        height=HEIGHT,
        dtype=rasterio.uint16,
        count=3)

    with rasterio.open('example.tif', 'w', **profile) as dst:
        dst.write(array)
