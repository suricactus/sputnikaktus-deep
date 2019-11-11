#!/usr/bin/env python3

from typing import (Union, Tuple, List, Optional)
from datetime import datetime
import argparse
import os
import json
import importlib.util


import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter
from utils import (PatchSize, NpEncoder, normalize_patch_size,
                   to_categorical_binary, fetch_images)
from visualization import (visualize_pairs)


def datasets(dataset, patch_size, classes):
    images_path, labels_path = dataset
    images = fetch_images(images_path, patch_size, bands=bands)
    labels = fetch_images(labels_path, patch_size, bands=(1,))
    labels = to_categorical_binary(labels, classes)

    return images, labels


def train_model(
    training: Tuple[str, str],
    bands: List[int],
    model: str,
    classes: int,
    patch_size: PatchSize,
    validation: Tuple[str, str] = None,
    test: Tuple[str, str] = None,
    weights: str = None,
    name: str = None,
    overwrite: bool = False,
    print_summary_only: bool = False,
    batch_size: int = None,
    epochs: int = 10,
    learning_rate: float = 0.001,
    interactive: bool = False
):
    model_path = model
    patch_size = normalize_patch_size(patch_size)
    name = name if name else datetime.now().strftime('%y%m%d_%H%M%S')

    assert isinstance(classes, int)
    assert classes > 0
    assert os.path.exists(model)

    spec = importlib.util.spec_from_file_location('model_module', model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model = model_module.get_model(
        patch_size,
        bands,
        classes,
        learning_rate
    )

    if print_summary_only:
      print(model.summary())
      exit(0)

    if weights:
        assert os.path.exists(weights)

        model.load_weights(weights)

    images_path_training, labels_path_training = training
    images_training = fetch_images(images_path_training, patch_size, bands=bands)
    labels_training = fetch_images(labels_path_training, patch_size, bands=(1,))

    if interactive:
        visualize_pairs(
            images_training,
            labels_training, 
            legend=('background', 'building'),
            index=0
        )

    labels_training = to_categorical_binary(labels_training, classes)

    print('Train shapes for images and labels: ', images_training.shape, labels_training.shape)

    history = model.fit(
        x=images_training,
        y=labels_training,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    weights_filename = os.path.join('experiment', name, 'model.h5')
    history_filename = os.path.join('experiment', name, 'history.json')

    os.makedirs(os.path.join('experiment', name), exist_ok=True)

    print('Saving model to "{}"...'.format(weights_filename))

    model.save(weights_filename)

    print('Saving history to "{}"...'.format(history_filename))

    with open(history_filename, 'w') as history_dest:
        data = vars(history)
        del data['model']
        json.dump(data, history_dest, cls=NpEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A tool to train deeplab v3+ model',
        epilog='A tool to train deeplab v3+ model'
    )
    parser.add_argument('--training', type=str, nargs=2, required=True,
                        help='Existing directory where the train images and labels reside')
    parser.add_argument('--validation', type=str, nargs=2,
                        help='Existing directory where the validate images and labels reside')
    parser.add_argument('--test', type=str, nargs=2,
                        help='Existing directory where the test images and labels reside')
    parser.add_argument('--name', type=str,
                        help='Directory where classfied images will be output. '
                        'If not provided, it is going to use the YYMMDD_HHMMSS format')
    parser.add_argument('--weights', type=str,
                        help='Pretrained weights to be used.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to be used.')
    parser.add_argument('--patch-size', type=int, nargs='+', required=True,
                        metavar='width height bands',
                        help='Size of the output input tensor.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs'
                        'Default: 10')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size')
    parser.add_argument('--bands', type=int, nargs='+', default=(1, 2, 3),
                        help='Bands to be be used.'
                        'Default: (1, 2, 3)')
    parser.add_argument('-c', '--classes', type=int, required=True,
                        help='Number of output classes.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete the whole destination path and then recreate it.'
                        'Default: False')
    parser.add_argument('--interactive', action='store_true',
                        help='Show plots and make the whole process interactive.'
                        'Default: False')
    parser.add_argument('--print-summary-only', action='store_true',
                        help='Print model summary and exit.'
                        'Default: False')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the network'
                        'Default: 0.001')

    args = parser.parse_args()

    train_model(**vars(args))
