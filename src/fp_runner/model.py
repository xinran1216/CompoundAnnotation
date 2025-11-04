
from tensorflow.keras.models import load_model
from .metrics import CUSTOM_OBJECTS

def load_keras_model(path: str):
    return load_model(path, custom_objects=CUSTOM_OBJECTS)
