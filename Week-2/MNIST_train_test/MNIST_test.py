#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

'''
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
'''

#讓tensorflow自行動態分配記憶體空間而不是強制佔用所有空間導致Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.compat.v1.Session(config=config)



# Read the .h5 file
model = load_model('MNIST_ep10.h5')

# Read the image
img = (cv2.imread('MNIST_test/1_draw.png', 0) / 225.0)
# Resize to 28x28x1
img = cv2.resize(img, (28,28))

# Reshape to 1x28x28x1 because of the input of network
img_reshaped = np.reshape(img, (1,28,28,1))
# Perdiction
output = model.predict(img_reshaped)
print('Ouput prob. result = ', output)
# Get the most probably number.
print('The probably number is ', np.argmax(output))
# Display the image you choose
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

