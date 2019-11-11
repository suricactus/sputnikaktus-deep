from typing import (Tuple, Iterable)

from tensorflow.keras.models import (Model, Sequential)
from tensorflow.keras.optimizers import (Adam, SGD)
from tensorflow.keras.layers import (ZeroPadding2D, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape)


def get_model(
    patch_size: Tuple[int, int],
    bands: Iterable[int],
    classes: int,
    learning_rate: float
) -> Model:
    img_w, img_h = patch_size
    input_shape = (img_w, img_h, len(bands))
    optimizer = SGD(lr=learning_rate, momentum=0.9)

    model = Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape=input_shape))
    model.add(Convolution2D(
        filters=32,
        kernel_size=(7, 7),
        dilation_rate=(1, 1)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1)
    ))
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(
        filters=32,
        kernel_size=(5, 5),
        dilation_rate=(1, 1)
    ))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1)
    ))
    model.add(Convolution2D(
        filters=classes,
        kernel_size=(1, 1)
    ))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy', 
        metrics=['accuracy'], 
        optimizer=optimizer
    )

    return model
