import json
from typing import (Any, Union, Tuple, List, Iterable, Iterator)
import importlib
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
from skimage import exposure

FilePath = str
ImageSize = Tuple[int, int]
PatchSize = Union[int, Tuple[int], Tuple[int, int]]
OverlapSize = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int, int]]
ClipExtremes = Union[int, Tuple[int, int]]


class PatchResidue(Enum):
    IGNORE = 'ignore'
    OVERLAP = 'overlap'


def hist_stretch_raster(img, bands: Iterable = None, clip_extremes: ClipExtremes = (2.5, 97.5)):
    """General purpose histogram stretching."""
    total_bands, *_dimensions = img.shape

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


def hist_stretch_image(img, bands: Iterable = None, clip_extremes: ClipExtremes = (2.5, 97.5)):
    """General purpose histogram stretching."""
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    print('shape', img.shape)

    assert len(img.shape) == 3, '(w, h, bands) image'

    _, _, total_bands = img.shape

    if bands is None:
        bands = range(0, total_bands)

    stretched = np.copy(img)

    if isinstance(clip_extremes, int):
        clip_extremes = (0 + clip_extremes, 100 - clip_extremes)

    for index, band in enumerate(bands):
        arr = img[:, :, band]
        percentile_min, percentile_max = np.percentile(arr, clip_extremes)
        band_stretched = exposure.rescale_intensity(
            arr, in_range=(percentile_min, percentile_max))

        stretched[:, :, index] = band_stretched

    stretched = np.squeeze(stretched)

    return stretched


def normalize_patch_size(patch_size: PatchSize) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if len(patch_size) == 1:
        patch_size = (patch_size[0], patch_size[0])

    # enforce to tuple
    patch_size = (patch_size[0], patch_size[1])

    assert isinstance(patch_size, tuple) and len(
        patch_size) == 2, 'tile_size must be a tuple of two elements'

    return patch_size


def normalize_overlap_size(overlap_size: OverlapSize) -> Tuple[int, int, int, int]:
    if isinstance(overlap_size, int):
        overlap_size = (overlap_size, overlap_size, overlap_size, overlap_size)

    if len(overlap_size) == 1:
        overlap_size = (overlap_size[0], overlap_size[0],
                        overlap_size[0], overlap_size[0])
    elif len(overlap_size) == 2:
        overlap_size = (overlap_size[0], overlap_size[1],
                        overlap_size[0], overlap_size[1])

    # enforce to tuple
    overlap_size = (overlap_size[0], overlap_size[1],
                    overlap_size[2], overlap_size[3])

    assert isinstance(overlap_size, tuple) and len(
        overlap_size) == 4, 'overlap_size must be a tuple of four elements'

    return overlap_size


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
    patch_residue: str = None,
    dtype: np.dtype = None
):
    assert dtype, 'Missing output datatype'

    patch_w, patch_height = normalize_patch_size(patch_size)
    files = get_filtered_files(path, filter)

    if len(files) == 0:
        raise Exception('no files match the provided filter "{}", exiting...'.format(str(filter)))

    patches_count = 0

    for filename in files:
        img_src: DatasetWriter
        with rio.open(filename, 'r') as img_src:
            offsets = get_patch_offsets(
                (img_src.meta['width'], img_src.meta['height']), patch_size, patch_residue)

            patches_count += len(list(offsets))

    # build the final result with the proper shape, so to prevent wasteful memory copies
    tiles = np.zeros(shape=(patches_count, patch_w, patch_height, len(bands)), dtype=dtype)
    index = 0

    for filename in files:
        img_src: DatasetWriter
        with rio.open(filename, 'r') as img_src:
            img_size = (img_src.meta['width'], img_src.meta['height'])
            for window in get_patch_windows(img_size, patch_size, patch_residue):
                img = img_src.read(
                    indexes=bands, window=window, out_dtype=dtype)

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
    patch_w, patch_height = normalize_patch_size(patch_size)

    col_offs = np.array(range(0, img_w, patch_w))
    row_offs = np.array(range(0, img_h, patch_height))

    if patch_w * len(col_offs) != img_w:
        if patch_residue is None or patch_residue == 'ignore':
            col_offs = col_offs[:-1]
        elif patch_residue == 'overlap':
            col_offs[-1] = img_w - patch_w
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


def get_patch_windows(img_size: ImageSize, patch_size: PatchSize, patch_residue: PatchResidue = None, img_transform=None) -> Iterator:
    img_w, img_h = img_size
    big_window = windows.Window(
        col_off=0, row_off=0, width=img_w, height=img_h)
    offsets = get_patch_offsets((img_w, img_h), patch_size, patch_residue)
    patch_w, patch_h = normalize_patch_size(patch_size)

    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=patch_w,
            height=patch_h
        ).intersection(big_window)

        if img_transform:
            transform = windows.transform(window, img_transform)
            yield window, transform
        else:
            yield window

def get_buffered_patch2(
    img: DatasetWriter,
    bands: Iterable[int],
    patch_size: PatchSize,
    overlap_size: PatchSize,
    patch_residue: PatchResidue = None,
    pad: str = None,
    out_dtype: np.dtype = np.uint8
) -> Iterator:
    img_w, img_h = (img.meta['width'], img.meta['height'])
    patch_w, patch_h = normalize_patch_size(patch_size)
    offsets = get_patch_offsets(
        (img_w, img_h), patch_size, patch_residue)
    overlap_t, overlap_r, overlap_b, overlap_l = normalize_overlap_size(
        overlap_size)
    big_window = windows.Window(
        col_off=0, row_off=0, width=img_w, height=img_h)

    for offset_col, offset_row in offsets:
        # real_overlap_l = max(0, min(overlap_l, offset_col - overlap_l))
        # real_overlap_t = max(0, min(overlap_t, offset_row - overlap_t))

        if overlap_l - offset_col >= 0:
            real_overlap_l = offset_col
        else:
            real_overlap_l = overlap_l

        if overlap_t - offset_row >= 0:
            real_overlap_t = offset_row
        else:
            real_overlap_t = overlap_t

        if offset_col + patch_w + overlap_r >= img_w:
            real_overlap_r = overlap_r - (offset_col + patch_w + overlap_r - img_w)
        else:
            real_overlap_r = overlap_r

        if offset_row + patch_h + overlap_b > img_h:
            real_overlap_b = overlap_b - (offset_row + patch_h + overlap_b - img_h)
        else:
            real_overlap_b = overlap_b

        window_w = min(img_w, patch_w + real_overlap_l + real_overlap_r)
        window_h = min(img_h, patch_h + real_overlap_t + real_overlap_b)

        window = windows.Window(
            col_off=offset_col - real_overlap_l,
            row_off=offset_row - real_overlap_t,
            width=window_w,
            height=window_h
        ).intersection(big_window)

        pad_size_t = overlap_t - real_overlap_t
        pad_size_b = overlap_b - real_overlap_b
        pad_size_l = overlap_l - real_overlap_l
        pad_size_r = overlap_r - real_overlap_r

        img_read = img.read(indexes=bands, window=window, out_dtype=out_dtype)

        padded_patch = np.pad(
            img_read,
            ((0, 0), (pad_size_t, pad_size_b), (pad_size_l, pad_size_r)),
            mode='constant',
            constant_values=0
        )

        yield reshape_as_image(padded_patch), offset_col, offset_row


# TODO actually returns Module class
def load_module(model_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(
            'model_module', model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    return model_module

# got it from https://stackoverflow.com/a/57915246
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
