import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from alibi.explainers import CounterFactual
import config
import preprocess as pp

tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
tf.compat.v1.disable_eager_execution()


def C_F(model_path, image_path):

    model = load_model(config.model_path)

    size = model.input_shape[1:3]

    image_array, array_shape = pp.get_img_array(image_path, size)

    image_array = np.expand_dims(image_array, axis=0)

    shape = image_array.shape

    # Counterfactual parameters
    target_proba = 1.0
    tol = 0.1 
    target_class = 'other' 
    max_iter = 10
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (image_array.min(),image_array.max())

    cf = CounterFactual(model, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)

    explanation = cf.explain(image_array)

    pred_class = explanation.cf['class']
    proba = explanation.cf['proba'][0][pred_class]

    print(f'Counterfactual prediction: {pred_class} with probability {proba}')

    plt.imshow(explanation.cf['X'].reshape(229,229,3))
    plt.show()


if __name__ == '__main__':

    counterfactual = C_F(config.model_path, config.image_path)


