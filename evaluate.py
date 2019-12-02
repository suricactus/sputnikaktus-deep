#!/usr/bin/env python3

from itertools import product

import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.plot import show

from sputnikaktus.utils import get_patch_offsets, get_buffered_patch2
from sputnikaktus.visualization import visualize_image
import matplotlib.pyplot as plt
from rasterio.plot import (reshape_as_image)


import rasterio
from rasterio.plot import show

# pyplot.imshow(src.read(1))
# pyplot.show()

# a = get_patch_offsets((321, 800), (150, 150), 'overlap')
from skimage import exposure

with rio.open('./data_mini/input/train/images/top_60cm_qb_area1_h1501_v1000_500x500.tif', 'r') as img_src:
    patch_size = (150, 150)
    overlap_size = (100, 100)

    img_patches = get_buffered_patch2(
        img_src, 
        bands=(1, 2, 3), 
        patch_size=patch_size, 
        overlap_size=overlap_size,
        patch_residue='overlap', 
        pad='zeros', 
        out_dtype=np.float32)
    img = img_src.read(indexes=(1, 2, 3), out_dtype=np.float32)

    visualize_image(img, as_image=True)
    img_output = reshape_as_image(np.zeros(shape=img.shape, dtype=np.float32))

    offset_t, offset_l = 0, 0

    for img_patch, offset_col, offset_row in img_patches:
        # print(img_patch, offset_col, offset_row)
        # _p stands for prediction
        padded_p = img_patch
        x_start = overlap_size[0]
        x_end = overlap_size[0] + patch_size[0]
        y_start = overlap_size[1]
        y_end = overlap_size[1] + patch_size[1]
        img_p = padded_p[x_start:x_end, y_start:y_end,:]

        print(offset_col, offset_col + patch_size[0], offset_row, offset_row + patch_size[1])

        img_output[
            offset_row:offset_row + patch_size[1], 
            offset_col:offset_col + patch_size[0],
            :
        ] = img_p

    visualize_image(img_output, show=True, stretch=False)
        


# print(list(a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A tool to train deeplab v3+ model',
        epilog='A tool to train deeplab v3+ model'
    )
    parser.add_argument('--training', type=str, nargs=2, required=True,
                        help='Existing directory where the train images and labels reside')
    parser.add_argument('--validation', type=str, nargs=2,
                        help='Existing directory where the validate images and labels reside')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to be used.')
    parser.add_argument('--patch-size', type=int, nargs='+', required=True,
                        metavar='width height',
                        help='Size of the output input tensor.')
    parser.add_argument('--weights', type=str,
                        help='Pretrained weights to be used.')
    parser.add_argument('--overlap-size', type=int, nargs='+', required=True,
                        metavar='width height',
                        help='Size of the overlap between patches.')
    parser.add_argument('--bands', type=int, nargs='+', default=(1, 2, 3),
                        help='Bands to be be used.'
                        'Default: (1, 2, 3)')
