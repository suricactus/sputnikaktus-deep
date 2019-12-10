#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from rasterio.profiles import DefaultGTiffProfile
import rasterio as rio
from PIL import Image


from model.model import Deeplabv3
from satellite.utils import (extract_dt, fetch_images, to_categorical_4d)


def evaluate(
    name: str,
    images: str,
    labels: str,
    weights: str,
    history: str
  ):
  dt = extract_dt(history)

  with open(history, 'r') as json_file:
    model_history = json.load(json_file)

    plt.plot(model_history['loss'])

    plt.title('{} ({}) training curve loss'.format(name, dt))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('{}_training_loss.png'.format(dt))

    plt.plot(1 - np.array(model_history['accuracy']))
    plt.title('{} ({}) training curve error rate'.format(name, dt))
    plt.ylabel('error rate')
    plt.xlabel('epoch')
    plt.savefig('{}_training_error_rate.png'.format(dt))

  tile_size = (512, 512)
  bands = (1, 2, 3)
  classes = 2
  input_shape = (*tile_size, len(bands))


  model = Deeplabv3(
    input_shape=input_shape,
    classes=classes,
    weights=None
  )
  model.load_weights(weights)
  optimizer = SGD(lr=0.001, momentum=0.9)
  model.compile(
      loss='binary_crossentropy',
      metrics=['accuracy'],
      optimizer=optimizer)
  # if print_summary_only:
  #   print(model.summary())
  #   exit

  images = fetch_images(images, bands=bands)
  labels = fetch_images(labels, bands=(1))

  images_test = np.zeros(shape=(0, tile_size[0], tile_size[1], len(bands)))
  labels_test = np.zeros(shape=(0, tile_size[0], tile_size[1], 1))

  for image in images:
    if image.shape[0] == tile_size[0] and image.shape[1] == tile_size[1]:
      image_tmp = np.expand_dims(image, axis=0)
      images_test = np.concatenate((images_test, image_tmp), axis=0)
    else:
      raise Exception('Not implemented yet')

  for image in labels:
    if image.shape[0] == tile_size[0] and image.shape[1] == tile_size[1]:
      image_tmp = np.expand_dims(image, axis=0)
      labels_test = np.concatenate((labels_test, image_tmp), axis=0)
    else:
      raise Exception('Not implemented yet')

  labels_test = to_categorical_4d(labels_test, classes)

  opt = model.evaluate(images_test, labels_test)
  predictions = model.predict(images_test)

  i = 1
  for labels in predictions:
    # prediction = np.stack(prediction, axis=0)

    labels = np.argmax(labels.squeeze(), -1)

    # print(resized_image.shape, np.expand_dims(resized_image, 0).shape)
    # remove padding and resize back to original image
    if 0 > 0:
        labels = labels[:-0]
    if 0 > 0:
        labels = labels[:, :-0]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((512, 512)))

    plt.imshow(labels)
    plt.waitforbuttonpress()


    # plt.imshow(prediction,  # cmap=cmap, norm=norm,
    #           interpolation='nearest', origin='upper')
    # foo = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 512, 'height': 512, 'count': 2, 'crs': 26986, 'tiled': False, 'interleave': 'pixel'}

    # with rio.open('output_{}.tif'.format(i), 'w', **foo) as dst_dataset:
    #   prediction = np.stack(prediction, axis=0)
    #   prediction[1:::]

    #   print(prediction.shape)
    #   print(prediction.astype(rio.uint8))

    #   dst_dataset.write(prediction.astype(rio.uint8), 1)

    # i += 1

  print('train and labels', images_test.shape, labels_test.shape)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='A tool to train deeplab v3+ model',
      epilog='A tool to train deeplab v3+ model'
  )
  parser.add_argument('-i', '--images', type=str, required=True,
                      help='Existing directory where the images reside')
  parser.add_argument('-l', '--labels', type=str, required=True,
                      help='Existing directory where the labeled images reside')
  parser.add_argument('--name', type=str, required=True, 
                      help='Name of the charts')
  parser.add_argument('--weights', type=str, required=True,
                      help='Pretrained weights to be used.')
  parser.add_argument('--history', type=str, required=True,
                      help='Pretrained history to be used.')

  args = parser.parse_args()

  evaluate(
    name=args.name,
    images=args.images,
    labels=args.labels,
    weights=args.weights,
    history=args.history,
  )
