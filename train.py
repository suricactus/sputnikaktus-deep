#!/usr/bin/env python3

from typing import (Union, Tuple, List, Optional)
from datetime import datetime
import argparse
import os
import json
import importlib.util


import numpy as np
from sputnikaktus.utils import (PatchSize, NumpyEncoder, normalize_patch_size,
                                to_categorical_binary, fetch_images, load_module)
from sputnikaktus.visualization import (visualize_pairs, show_history)

def preprocess(images: np.array, labels: np.array):
    labels[labels != 1] = 2
    labels = labels - 1

    return images, labels


def preprocess_wrapper(images: np.array, labels: np.array, classes: int):
    images, labels = preprocess(images, labels)

    assert len(np.unique(labels)) == classes, 'The number of unique values in the labels should match exactly the number of classes'
    assert np.min(labels) == 0, 'The values in labels should start from 0'
    assert np.max(labels) == classes - 1, 'The values in labels should not exceed the classes'

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

    model_module = load_module(model_path)

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
    images_training = fetch_images(images_path_training, patch_size, bands=bands, dtype=np.float32)
    labels_training = fetch_images(labels_path_training, patch_size, bands=(1,), dtype=np.int8)
    images_training, labels_training = preprocess_wrapper(images_training, labels_training, classes)

    validation_data = None
    if validation:
        images_path_validation, labels_path_validation = validation
        images_validation = fetch_images(images_path_validation, patch_size, bands=bands, dtype=np.float32)
        labels_validation = fetch_images(labels_path_validation, patch_size, bands=(1,), dtype=np.int8)
        images_validation, labels_validation = preprocess_wrapper(images_validation, labels_validation, classes)
        labels_validation = to_categorical_binary(labels_validation, classes)
        validation_data = (images_validation, labels_validation)

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
        validation_data=validation_data
    )

    weights_filename = os.path.join('experiment', name, 'model.h5')
    history_filename = os.path.join('experiment', name, 'history.json')
    history_plot_filename = os.path.join('experiment', name, 'history.png')

    os.makedirs(os.path.join('experiment', name), exist_ok=True)

    print('Saving model to "{}"...'.format(weights_filename))

    model.save(weights_filename)

    print('Saving history to "{}"...'.format(history_filename))
    print('Saving history to "{}"...'.format(history_plot_filename))

    with open(history_filename, 'w') as history_dest:
        data = vars(history)
        del data['model']
        json.dump(data, history_dest, cls=NumpyEncoder)

    plt = show_history(name, history.history, history.params)
    plt.savefig(history_plot_filename)

    if interactive:
        plt.show()


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
                        metavar='width height',
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
