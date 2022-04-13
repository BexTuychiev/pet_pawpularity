import logging
import warnings

import numpy as np
import bentoml
from bentoml.io import Text
from skimage.transform import resize


def create_bento_service_keras(bento_name):
    """
    Create a Bento service for a Keras model.
    """
    # Load the model
    keras_model = bentoml.keras.load_runner(bento_name)

    # Create the service
    service = bentoml.Service(bento_name + "_service", runners=[keras_model])

    return keras_model, service



