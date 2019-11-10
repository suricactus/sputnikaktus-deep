from typing import (Union, Tuple, List, Iterable, Iterator)
import os
from glob import glob
from itertools import product
from enum import Enum


import numpy as np
import rasterio as rio
from deprecated import deprecated
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


# TODO check if keras to_categorical can do the job
def to_categorical_binary(images, classes):
    # TODO this is binary only :(
    class_values = np.where(images != 0, 1, 0)
    # (0, 1, 0, 1, 0, 0) -> ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (1, 0))
    with_categorical = np.eye(classes, dtype=int)[class_values]
    # (n, w, h, 1, c) -> (n, w, h, c)
    categorical_labels = np.squeeze(with_categorical)

    return categorical_labels


@deprecated('Please use to_categorical_binary')
def to_categorical_4d(images, classes):
    (num_samples, img_w, img_h, _) = images.shape
    # a = categorical_labels
    Y = np.zeros((num_samples, img_w, img_h, classes), dtype=int)
    for h in range(images.shape[0]):
        for i in range(images.shape[1]):
            for j in range(images.shape[2]):
                if images[h, i, j, 0] != 0:
                    Y[h, i, j, 1] = 1
                else:
                    continue

    # for h in range(images.shape[0]):
    #     for i in range(images.shape[1]):
    #         for j in range(images.shape[2]):
    #             if (Y[h, i, j, 0] != a[h, i, j, 0] or Y[h, i, j, 1] != a[h, i, j, 1]):
    #                 print(h, i, j, images[h, i, j, 0], Y[h, i, j, 0], Y[h, i, j, 1], a[h, i, j, 0], a[h, i, j, 1])

    print(np.array_equal(a, Y))

    return Y


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
    as_image: bool = True,
    patch_residue: str = None
):
    patch_width, patch_height = normalize_patch_size(patch_size)
    files = get_filtered_files(path, filter)

    if len(files) == 0:
        print('no files match the provided filter "{}", exiting...'.format(str(filter)))
        exit(1)

    patches_count = 0

    for filename in files:
        img_src: DatasetWriter
        with rio.open(filename, 'r') as img_src:
            offsets = get_patch_offsets(
                (img_src.meta['width'], img_src.meta['height']), patch_size, patch_residue)

            patches_count += len(list(offsets))

    # build the final result with the proper shape, so to prevent wasteful memory copies
    tiles = np.zeros(
        shape=(patches_count, patch_width, patch_height, len(bands)))
    index = 0

    for filename in files:
        img_src: DatasetWriter
        with rio.open(filename, 'r') as img_src:
            for window, transform in get_patch_windows(img_src, patch_size, patch_residue):
                img = img_src.read(indexes=bands, window=window)

                if np.ndim(img) == 2:
                    img = np.expand_dims(img, axis=0)

                if as_image:
                    # an array in shape (width, height, bands)
                    img = reshape_as_image(img)

                img = np.expand_dims(img, axis=0)
                tiles[index] = img
                index += 1

    return tiles


def get_patch_offsets(image_size: Tuple[int, int], patch_size: PatchSize, patch_residue: PatchResidue):
    img_w, img_h = image_size
    patch_width, patch_height = normalize_patch_size(patch_size)

    col_offs = np.array(range(0, img_w, patch_width))
    row_offs = np.array(range(0, img_h, patch_height))

    if patch_width * len(col_offs) != img_w:
        if patch_residue is None or patch_residue == 'ignore':
            col_offs = col_offs[:-1]
        elif patch_residue == 'overlap':
            col_offs[-1] = img_w - patch_width
        else:
            pass

    if patch_height * len(row_offs) != img_h:
        if patch_residue is None or patch_residue == 'ignore':
            row_offs = col_offs[:-1]
        elif patch_residue == 'overlap':
            row_offs[-1] = img_h - patch_height
        else:
            pass

    return product(
        col_offs,
        row_offs
    )


def get_patch_windows(img: DatasetWriter, patch_size: PatchSize, patch_residue: PatchResidue = None) -> Iterator:
    offsets = get_patch_offsets(
        (img.meta['width'], img.meta['height']), patch_size, patch_residue)
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
