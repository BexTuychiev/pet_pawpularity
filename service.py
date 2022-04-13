import logging
import warnings

import numpy as np
import bentoml
from bentoml.io import Text, NumpyNdarray
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


model, service = create_bento_service_keras("conv2d_larger_dropout")


# Create an API function
@service.api(input=Text(), output=NumpyNdarray())
def predict(image_str) -> np.ndarray:
    """
    Predict pet pawpularity from an image using the given Bento.
    """
    # Convert the image back to numpy array
    image = np.fromstring(image_str, np.uint8)
    image = resize(image, (224, 224, 3))
    image = image / 255.0

    result = model.run(image)

    return result
