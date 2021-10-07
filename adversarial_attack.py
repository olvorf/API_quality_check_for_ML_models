from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import matplotlib.pyplot as plt
import preprocess as pp
import config
from tqdm import tqdm

tf.compat.v1.disable_eager_execution()



def create_adversarial(image_path, model_path, eps):
    
    
    model = load_model(model_path)
    size =  model.input_shape[1:3]

    image = keras.preprocessing.image.load_img(image_path, target_size=size)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    #image = tf.constant(image, dtype=tf.float32)
    image_node = tf.compat.v1.placeholder(
        dtype=tf.float32,
        shape=image.shape
    )



    
    prediction = model(image_node)
    input_label = tf.argmax(prediction,axis=1)
    loss = sparse_categorical_crossentropy(input_label, prediction)
    
    gradient = tf.gradients(ys=loss, xs=image_node)
    initialize = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as session:

        session.run(initialize)
        _gradient = session.run(
                gradient,
                feed_dict={
                    image_node: image
                }
            )
        gradient_step = np.sign(_gradient[0]) 
        adv_image = image - eps*gradient_step
    
    adv_image = np.clip(adv_image, -1, 1)
    
    prediction, prediction_label = pp.make_predictions(model, adv_image)

    plt.title('Class {} with {:.2f}% Probability'.format(prediction_label, np.max(prediction*100)))
    plt.imshow(adv_image[0])
    plt.show()


if __name__ == '__main__':
    adv = create_adversarial(config.image_path, config.model_path, 0.04)




    

 

