
import tensorflow as tf
import tensorflow_transform as tft

import numpy as np

import base64

import multimodal_constants

_IMAGE = multimodal_constants.IMAGE
_QUERY = multimodal_constants.QUERY
_PRODUCT = multimodal_constants.PRODUCT
_LABEL_KEY = multimodal_constants.LABEL_KEY
_transformed_name = multimodal_constants.transformed_name

def preprocessing_fn(inputs):

    features_dict = {}

    ### START CODE HERE ###

    ### Decode boxes BYTES to tensor
    # print(inputs[_IMAGE[3]])
    # boxes = tf.io.decode_base64(inputs[_IMAGE[3]])
    # print(boxes)
    # boxes = tf.frombuffer(tf.io.decode_raw(boxes, tf.float32))
    # print(boxes)
    y = tf.io.decode_base64(inputs[_IMAGE[3]], pad=True)
    # z = tf.compat.bytes_or_text_types(inputs[_IMAGE[3]])
    print(y)
    # x = np.frombuffer(base64.b64decode(), dtype=np.float32).reshape(inputs[_IMAGE[2]], 4)
    # print(x)
    # features_dict[_transformed_name(_IMAGE[3])] = 
    # print(features_dict[_transformed_name(_IMAGE[3])])

    
    ### END CODE HERE ###  

    # No change in the label
    # features_dict[_LABEL_KEY] = inputs[_LABEL_KEY]

    return features_dict
