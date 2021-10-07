import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import preprocess_input

from keras.models import load_model

import config



def image_data_preprocess(image_data, size):

    ''' 
    Parameters
    -----------
    image data: image path
    size: image size 


    Return
    -----------
    Image data generator:
    Image classes:
    Image class labels


    '''
    
    image_datagen = ImageDataGenerator(rescale=1./255)

    image_generator = image_datagen.flow_from_directory(
            image_data,
            class_mode="categorical",
            target_size=size,
            color_mode="rgb",
            shuffle=False,
            batch_size=32)
    
    image_classes = image_generator.classes

    image_classes_name = image_generator.class_indices.keys()


    return image_generator, image_classes, image_classes_name


def make_predictions(model, image_generator):

    ''' 
    Parameters
    -----------
    model: keras model  " *.h5 "
    image data: 4d numpy array
    
    
    
    Return
    -----------
    prediction:
    prediction class: 
    
    '''


    predictions = model.predict(image_generator, steps=len(image_generator), verbose=1)
    predictions_idxs = np.argmax(predictions, axis=1)


    return  predictions, predictions_idxs, 

def get_img_array(image_path, size):

    ''' 
    Parameters
    ------------
    image_path: Image path
    size: size of the image 



    Return
    -----------
    image_array: image array
    array_shape: image array shape
    '''

    
    image = keras.preprocessing.image.load_img(image_path, target_size=size)
    
    image_array = keras.preprocessing.image.img_to_array(image)

    # image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    array_shape = image_array.shape

    return image_array, array_shape


#if __name__ == '__main__':





