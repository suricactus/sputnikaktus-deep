#!/usr/bin/env python3

from typing import (Union, Tuple, List, Optional)
from datetime import datetime
import argparse
import os
import json

import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter
from tensorflow.keras.optimizers import SGD
from utils import (PatchSize, normalize_patch_size, to_categorical_4d, fetch_images)
from model.model import Deeplabv3


def train_model(
    images_path: str,
    labels_path: str,
    # destination: str,
    tile_size: PatchSize,
    classes: int,
    bands: List[int],
    weights: str = None,
    overwrite: bool = False,
    print_summary_only: bool = False,
    verbose: bool = False,
    batch_size: int = None,
    epochs: int = 10,
    learning_rate: int = 0.001
  ):
  tile_size = normalize_patch_size(tile_size)
  input_shape = (*tile_size, len(bands))
  
  # model = Deeplabv3(
  #     input_shape=input_shape,
  #     classes=classes,
  #     weights=None
  #   )

  # if weights:
  #   model.load_weights(weights)

  # if print_summary_only:
  #   print(model.summary())
  #   exit

  # if verbose:
  #   print(model.summary())

  images = fetch_images(images_path, tile_size, bands=bands)
  labels = fetch_images(labels_path, tile_size, bands=(1,))

  print(images.shape)

  # optimizer = SGD(lr=learning_rate, momentum=0.9)
  # model.compile(
  #     loss='binary_crossentropy',
  #     metrics=['accuracy'], 
  #     optimizer=optimizer)

  # images_train = np.zeros(shape=(0, tile_size[0], tile_size[1], len(bands)))
  # labels_train = np.zeros(shape=(0, tile_size[0], tile_size[1], 1))

  # for image in images:
  #   if image.shape[0] == tile_size[0] and image.shape[1] == tile_size[1]:
  #     image_tmp = np.expand_dims(image, axis=0)
  #     images_train = np.concatenate((images_train, image_tmp), axis=0)
  #   else:
  #     raise Exception('Not implemented yet')

  # for image in labels:
  #   if image.shape[0] == tile_size[0] and image.shape[1] == tile_size[1]:
  #     image_tmp = np.expand_dims(image, axis=0)
  #     labels_train = np.concatenate((labels_train, image_tmp), axis=0)
  #   else:
  #     raise Exception('Not implemented yet')
      
  # labels_train = to_categorical_4d(labels_train, classes)
    
  # print('train and labels', images_train.shape, labels_train.shape)
  
  # history = model.fit(
  #     x=images_train,
  #     y=labels_train,
  #     epochs=epochs,
  #     batch_size=batch_size,
  #     verbose=2,
  # )

  # dt = datetime.now().isoformat()
  # weights_filename = 'model_{}.h5'.format(dt)
  # history_filename = 'history_{}.json'.format(dt)

  # print('Saving model to "{}"...'.format(weights_filename))

  # model.save(weights_filename)

  # print('Saving history to "{}"...'.format(history_filename))

  # with open(history_filename, 'w') as history_dest:
  #   json.dump(str(history.history), history_dest)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='A tool to train deeplab v3+ model',
      epilog='A tool to train deeplab v3+ model'
  )
  parser.add_argument('-i', '--images', type=str, required=True,
                      help='Existing directory where the images reside')
  parser.add_argument('-l', '--labels', type=str, required=True,
                      help='Existing directory where the labeled images reside')
  # parser.add_argument('-d', '--destination', type=str, required=True,
  #                     help='Directory where classfied images will be output')
  parser.add_argument('-t', '--tile-size', type=int, nargs='+', required=True,
                      metavar='width height bands',
                      help='Size of the output input tensor.')
  parser.add_argument('--batch-size', type=int,
                      help='Batch size')
  parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs'
                        'Default: 10')
  parser.add_argument('-b', '--bands', type=int, nargs='+', default=(1, 2, 3),
                      help='Bands to be be used to train the input.'
                        'Default: (1, 2, 3)')
  parser.add_argument('-c', '--classes', type=int, required=True,
                      help='Number of output classes.')
  parser.add_argument('-w', '--weights', type=str,
                      help='Pretrained weights to be used.')
  parser.add_argument('--overwrite', action='store_true',
                      help='Delete the whole destination path and then recreate it.'
                        'Default: False')
  parser.add_argument('--verbose', action='store_true',
                      help='High verbosity.'
                        'Default: False')
  parser.add_argument('--print-summary-only', action='store_true',
                      help='Print model summary and exit.'
                        'Default: False')
  parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for the network'
                        'Default: 0.001')

  args = parser.parse_args()

  train_model(
      images_path=args.images,
      labels_path=args.labels,
      # destination=args.destination,
      tile_size=args.tile_size,
      classes=args.classes,
      bands=args.bands,
      weights=args.weights,
      overwrite=args.overwrite,
      print_summary_only=args.print_summary_only,
      verbose=args.verbose,
      epochs=args.epochs,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate
  )
