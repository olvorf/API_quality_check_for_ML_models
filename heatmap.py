import numpy as np
import tensorflow as tf
from tensorflow import keras
import eli5
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import config
import matplotlib.pyplot as plt


tf.compat.v1.disable_eager_execution()

import preprocess as pp


def get_heatmap(model_path, image_path):


    model = load_model(config.model_path)

    size = model.input_shape[1:3]

    image_array, array_shape = pp.get_img_array(image_path, size)

    image_array = np.expand_dims(image_array, axis=0)

    explanation = eli5.explain_prediction(model, image_array)

    image_heatmap = eli5.format_as_image(explanation)

    print(explanation)

    plt.imshow(image_heatmap)
    plt.show()

if __name__ == '__main__':

    heatmap = get_heatmap(config.model_path, config.image_path)