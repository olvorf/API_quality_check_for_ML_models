import numpy as np
from IPython.display import display
from PIL import Image
import tensorflow as tf
import keras
import eli5
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from alibi.explainers import AnchorImage
import matplotlib
import matplotlib.pyplot as plt
import preprocess as pp
import config


tf.compat.v1.disable_eager_execution()
    

def anchor_generator(model_path, image_path):

    model = load_model(config.model_path)

    size = model.input_shape[1:3]

    image_array, array_shape = pp.get_img_array(image_path, size)

    predict_fn = lambda x: model.predict(x)

    segmentation_fn = 'slic'
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}

    explainer = AnchorImage(predict_fn, array_shape, segmentation_fn=segmentation_fn,
                        segmentation_kwargs=kwargs, images_background=None)

    explanation = explainer.explain(image_array, threshold=.95, p_sample=.5, tau=0.25)
    plt.imshow(explanation['anchor'])
    plt.show()


if __name__ == '__main__':

    anchor = anchor_generator(config.model_path, config.image_path)







