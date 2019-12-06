#!/usr/bin/env python3

from typing import (Union, Tuple, List, Iterable, Iterator)
from itertools import product
import argparse

import numpy as np
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.plot import reshape_as_image

from sputnikaktus.utils import (get_buffered_patch2, get_filtered_files, normalize_overlap_size, load_module,
                                PatchSize, OverlapSize, NumpyEncoder, normalize_patch_size, get_patch_offsets)
from sputnikaktus.visualization import visualize_image


def evaluate_model(
    evaluation: Tuple[str, str],
    model: str,
    weights: str,
    patch_size: PatchSize,
    overlap_size: OverlapSize,
    bands: Iterable[int]
) -> None:
    images_path_evaluation, labels_path_evaluation = evaluation
    image_filenames = get_filtered_files(images_path_evaluation, filter=('*.tif', '*.tiff'))
    label_filenames = get_filtered_files(labels_path_evaluation, filter=('*.tif', '*.tiff'))
    patch_w, patch_h = normalize_patch_size(patch_size)
    overlap_t, overlap_r, overlap_b, overlap_l = normalize_overlap_size(overlap_size)
    patch_padded_w = patch_w + overlap_l + overlap_r
    patch_padded_h = patch_h + overlap_t + overlap_b

    model_module = load_module(model)
    model = model_module.get_model(
        patch_size=(patch_padded_w, patch_padded_h), 
                    bands=bands, 
                    classes=2)
    model.load_weights(weights)

    for image_filename in image_filenames:
        with rio.open(image_filename, 'r') as img_src:
            img_patches = get_buffered_patch2(
                img_src,
                bands=bands,
                patch_size=patch_size,
                overlap_size=overlap_size,
                patch_residue='overlap',
                pad='zeros',
                out_dtype=np.float32)
            img = img_src.read(indexes=(1, 2, 3), out_dtype=np.float32)

            visualize_image(img, as_image=True, title='Raw image')
            img_output = np.zeros(shape=(2001, 2001), dtype=np.float32)

            for img_patch, offset_col, offset_row in img_patches:
                img_patch = np.expand_dims(img_patch, axis=0)
                padded_p = model.predict(img_patch)[0]

                x_start, x_end = (overlap_l, overlap_r + patch_w)
                y_start, y_end = (overlap_t, overlap_b + patch_h)
                img_p = padded_p[x_start:x_end, y_start:y_end, :]

                img_output[
                    offset_row:offset_row + patch_h,
                    offset_col:offset_col + patch_w
                ] = np.argmax(img_p, axis=2)


            visualize_image(img_output, show=True, stretch=True, title='Prediction')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A tool to train deeplab v3+ model',
        epilog='A tool to train deeplab v3+ model'
    )

    parser.add_argument('--evaluation', type=str, nargs=2, required=True,
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

    args = parser.parse_args()

    evaluate_model(**vars(args))
