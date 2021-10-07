import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from numpy import asarray
import cv2
import config

tf.compat.v1.disable_eager_execution()



def image_smoother(
        input_image: np.ndarray,
        kernel_height=5,
        kernel_width=5,
        stride_height=1,
        stride_width=1) -> np.ndarray:

    '''
    Takes an image and optimizes it s.t. the each pixel in the local neigbourhood is equal
    to the mean value of pixels in its local neihbourhood.
    
    
    Parameters: 
    -----------------
    input_image: 4d numpy array 
    kernel_height: pixel height of the kernel (odd number)
    kernel_width: pixel width of the kernel (odd number)
    stride_height: 
    stride_width: 


    Return: 
    ------------------
    processed image - numpy array with same dimension as input
                    - dtype:'float32'

    '''
    assert input_image.ndim == 4
    assert input_image.shape[0] == 1 and input_image.shape[-1] == 3

    input_node = tf.compat.v1.placeholder(
    dtype=tf.float32,
    shape=input_image.shape
    )
    


    # Create a Box Blur filter 
    kernel = tf.constant(1/(kernel_height*kernel_width), shape=(kernel_height,kernel_width,3,1),dtype=tf.float32)
    
    # Input image tensor
    image = tf.constant(input_image, shape=input_image.shape, dtype=tf.float32)

    # Convolve the input image with the kernel to obtain the blurred image. 
    blur_image = tf.nn.depthwise_conv2d(image, kernel, strides=[1,stride_height,stride_width,1], padding='SAME')
    
    # The blurred image has the same size as the input image adn the ojective is to minimize
    # the difference between the blurred image and the original one
    objective = tf.square(input_node - blur_image)


    gradient_node = tf.gradients(ys=objective, xs=input_node)

    with tf.compat.v1.Session() as session:
        for _ in tqdm(range(1000), desc="optimize image"):
            _gradient = session.run(
                gradient_node,
                feed_dict={
                    input_node: input_image
                }
            )
            gradient_step = np.sign(_gradient[0]) * (1 / 255)
            input_image = np.clip(input_image - gradient_step, 0, 255)
    return input_image

if __name__ == '__main__':
    
    #load image as array
    image_path = config.image_path
    image = Image.open(image_path)
    image_array = asarray(image)
    image_array = np.expand_dims(image_array, axis=0)

    # image smoother 
    new_image = image_smoother(image_array, kernel_height=5, kernel_width=5,stride_height=1,stride_width=1)
    
    # change the array type to 'uint8' and visualize it 
    new_image = new_image.astype(np.uint8)
    new_image = new_image[0,:,:,:]
    img = Image.fromarray(new_image, 'RGB')
    img.show()