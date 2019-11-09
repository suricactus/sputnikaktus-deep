from typing import (Union, Tuple, List, Iterable, Iterator)
import os
from glob import glob
from itertools import product
from enum import Enum

import numpy as np
import rasterio as rio
from rasterio import (windows)
from rasterio.plot import (reshape_as_image)
from rasterio.io import DatasetWriter

PatchSize = Union[int, Tuple[int], Tuple[int, int]]
ClipExtremes = Union[int, Tuple[int, int]]

class PatchResidue(Enum):
  IGNORE = 'ignore'
  OVERLAP = 'overlap'


def hist_stretch(img, bands: Iterable = None, clip_extremes: ClipExtremes = (0, 100)):
    """General purpose histogram stretching."""
    total_bands, *dimensions = img.shape

    if bands is None:
        bands = range(1, total_bands)

    stretched = []

    if isinstance(clip_extremes, int):
        clip_extremes = (0 + clip_extremes, 100 - clip_extremes)

    for band in bands:
        arr = img[band]
        percentile_min, percentile_max = np.percentile(arr, clip_extremes)
        img_rescale = exposure.rescale_intensity(
            arr, in_range=(percentile_min, percentile_max))

        stretched.append(img_rescale)

    return np.array(stretched)


def normalize_patch_size(patch_size: PatchSize) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if len(patch_size) == 1:
        patch_size = (patch_size[0], patch_size[0])

    # enforce to tuple
    patch_size = (patch_size[0], patch_size[1])

    if not isinstance(patch_size, tuple) or len(patch_size) != 2:
        raise Exception('tile_size must be a tuple of two elements')

    return patch_size


def get_filtered_files(
    path: str,
    filter: Union[str, Iterable[str]],
    recursive: bool = True
) -> List[str]:
    files = []

    if isinstance(filter, str):
        filter = (filter)

    for ext in filter:
        middle = '**' if recursive else ''
        files.extend(
            glob(os.path.join(path, middle, ext), recursive=recursive))

    return sorted(files)


def to_categorical_4d(images, classes):
    Y = np.zeros((
        images.shape[0],
        images.shape[1],
        images.shape[2],
        classes
    ), dtype=np.int32)

    for h in range(images.shape[0]):
        for i in range(images.shape[1]):
            for j in range(images.shape[2]):
                if images[h, i, j, 0] != 0:
                    Y[h, i, j, 1] = 1
                else:
                    continue

    return Y


def extract_dt(filename: str) -> str:
    basename_ext = os.path.basename(filename)
    basename, _ = os.path.splitext(basename_ext)
    _, dt = basename.split('_')

    return dt


def get_confusion_matrix(predictions, labels):
    classes = np.unique(labels)
    nbclasses = classes.size

    assert labels.size != predictions.size, 'There should be the same number of predictions and labels.'

    merged = np.concatenate((
        predictions.reshape(predictions.size, 1),
        labels.reshape(labels.size, 1)
    ), axis=1)

    CM = np.zeros((classes[-1] + 1, classes[-1] + 1))

    for class1 in classes:
        for class2 in classes:
            CM[class1, class2] = np.sum(np.logical_and(
                merged[:, 1] == class1, merged[:, 0] == class2))

    return CM


def fetch_images(
    path: str,
    patch_size: PatchSize,
    bands: List[int] = (1, 2, 3),
    filter: List[str] = ('*.tif', '*.tiff'),
    as_image: bool = True
):
    patch_width, patch_height = normalize_patch_size(patch_size)
    tiles = np.zeros(shape=(0, patch_width, patch_height, len(bands)))
    files = get_filtered_files(path, filter)

    if len(files) == 0:
        print('no files match the provided filter "{}", exiting...'.format(str(filter)))
        exit(1)

    for filename in files:
        img_src: DatasetWriter
        with rio.open(filename, 'r+') as img_src:
            img = img_src.read(indexes=bands)

            print(img.shape)
            if np.ndim(img) == 2:
                img = np.expand_dims(img, axis=0)

            if as_image:
                # an array in shape (width, height, bands)
                img = reshape_as_image(img)
            print(img.shape)

            img = np.expand_dims(img, axis=0)
            np.concatenate((tiles, img), axis=0)

    return tiles


def get_patch_offsets(image_size: Tuple[int, int], patch_size: PatchSize, patch_residue: PatchResidue):
    img_width, img_height = image_size
    patch_width, path_height = normalize_patch_size(patch_size)

    col_offs = np.array(range(0, img_width, patch_width))
    row_offs = np.array(range(0, img_height, path_height))

    if patch_width * len(col_offs) != img_width:
        if patch_residue == PatchResidue.OVERLAP:
            col_offs[-1] = img_width - patch_width
        elif patch_residue is PatchResidue.IGNORE:
            col_offs = col_offs[:-1]
        else:
            pass

    if path_height * len(row_offs) != img_height:
        if patch_residue == PatchResidue.OVERLAP:
            row_offs[-1] = img_height - path_height
        elif patch_residue is PatchResidue.IGNORE:
            row_offs = col_offs[:-1]
        else:
            pass

    return product(
        col_offs,
        row_offs
    )


def get_patch_windows(img: DatasetWriter, patch_size: PatchSize, rest: PatchResidue = None) -> Iterator:
    offsets = get_patch_offsets(
        (img.meta['width'], img.meta['height']), patch_size, rest)
    patch_width, path_height = normalize_patch_size(patch_size)

    big_window = windows.Window(
        col_off=0,
        row_off=0,
        width=img.meta['width'],
        height=img.meta['height'])

    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=patch_width,
            height=path_height
        ).intersection(big_window)

        transform = windows.transform(window, img.transform)
        yield window, transform
