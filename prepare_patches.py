#!/usr/bin/env python3

from typing import (Tuple, List, Iterator, Any)
import sys
import os
import pathlib
import argparse
import shutil

import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio import windows
from utils import (PatchSize, PatchResidue, normalize_patch_size,
                   get_filtered_files, get_patch_windows)


def get_dest_filename(dest_filename: str, *args: List[Any]) -> str:
  dest_filename_tmpl, dest_ext = os.path.splitext(dest_filename)

  if dest_ext in {'.tif', '.tiff'}:
    dest_ext = '.tif'

  dest_filename_tmpl += '_h{}_v{}_{}x{}%s' % dest_ext

  return dest_filename_tmpl.format(*args)


def write_tiles(
    source: str,
    destination: str,
    patch_size: PatchSize,
    patch_residue: PatchResidue,
    filter: List[str] = ('*.tif', '*.tiff'),
    overwrite: bool = False
) -> None:
  assert isinstance(source, str), 'source path must be a string'
  assert isinstance(destination, str), 'destination path must be a string'
  assert source != destination, 'source and destination must be different'
  assert os.path.isdir(source), 'source path must be an existing directory'
  assert len(patch_size) in {
      1, 2}, 'tile size must be either single or two integers'

  files = get_filtered_files(source, filter)

  if len(files) == 0:
    raise Exception('no files match the provided filter "{}", exiting...'.format(str(filter)))

  if overwrite:
    if os.path.exists(destination):
      shutil.rmtree(destination)

  assert not os.path.exists(destination), 'destination should not exist'

  for filename in files:
      print('Working on file file: ', filename)

      relative_filename = filename.replace(source, '', 1)[1:]
      dest_dirname = os.path.join(
          destination, os.path.dirname(relative_filename))

      print('Ensuring directory exists: ', dest_dirname)
      pathlib.Path(dest_dirname).mkdir(parents=True, exist_ok=True)

      img_src: DatasetWriter
      with rio.open(filename, 'r+') as img_src:
          meta = img_src.meta.copy()

          for window, transform in get_patch_windows(img_src, patch_size, patch_residue):
              meta['transform'] = transform
              meta['width'], meta['height'] = window.width, window.height
              dest_filename = get_dest_filename(
                  os.path.join(destination, relative_filename),
                  int(window.col_off),
                  int(window.row_off),
                  window.width,
                  window.height
              )

              print('Writing file: ', dest_filename)

              with rio.open(dest_filename, 'w', **meta) as img_dest:
                  img_dest.write(img_src.read(window=window))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A tool to support preparing datasets for machine learning.'
        'Walks over a given path, gets all the images that match a given filter'
        'and finally recreates the same filesystem tree in another specified location',
        epilog='Tiling images into desired shape in a parallel filesystem tree'
    )
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Existing directory path where the files are located')
    parser.add_argument('-d', '--destination', type=str, required=True,
                        help='Directory where to store the tiled images')
    parser.add_argument('-p', '--patch-size', type=int, nargs='+', required=True, metavar='width height',
                        help='Size of the output patches. If single integer provided, then '
                        'it the output tiles are squares with that size. If two values '
                        'provided, then the output patches are rectangles with the first '
                        'value representing the width is created')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete the whole destination path and then recreate it.'
                        'Default: False')
    parser.add_argument('--filter', type=str, nargs='*', default=('*.tif', '*.tiff'),
                        help='Glob patterns to filter files to be processed.')
    parser.add_argument('--patch-residue', type=str, choices=('overlap', 'ignore'), default='overlap',
                        help='What to do when there are no cropped areas of the image.'
                        'Default: overlap')

    args = parser.parse_args()

    write_tiles(
        source=args.source,
        destination=args.destination,
        patch_size=args.patch_size,
        overwrite=args.overwrite,
        filter=args.filter,
        patch_residue=args.patch_residue
    )
