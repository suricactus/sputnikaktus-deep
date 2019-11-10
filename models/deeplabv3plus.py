from typing import (Tuple, Iterable)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def get_model(
    patch_size: Tuple[int, int],
    bands: Iterable[int],
    classes: int,
    learning_rate: float
) -> Model:
    optimizer = Adam(learning_rate)
    model = Model()

    model.compile(
        optimizer=optimizer
    )

    return model
