from tensorflow.keras.models import load_model as _load_model
from .metrics import CUSTOM_OBJECTS


def load_keras_model(path: str):
    return _load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
