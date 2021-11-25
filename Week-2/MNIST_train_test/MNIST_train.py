#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout,Dense,Flatten,Conv2D,AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D,Softmax
import matplotlib.pyplot as plt
import numpy as np


# In[2]:
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


#將MNIST 手寫數字資料讀進來
mnist = tf.keras.datasets.mnist
# mnist 的load_data()會回傳已經先分割好的training data 和 testing data
# 並且將每個 pixel 的值從 Int 轉成 floating point 同時做normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('Loading finished')


#讓tensorflow自行動態分配記憶體空間而不是強制佔用所有空間導致Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.compat.v1.Session(config=config)

# In[3]:


x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))


# In[4]:


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5),strides=(1,1), activation='tanh'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=10, kernel_size=(3,3), strides=(1,1), padding='same',activation='tanh'))
model.add(Flatten())
model.add(Dense(units=120, activation='tanh'))
model.add(Dense(units=84, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))


# model每層定義好後需要經過compile
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[5]:


# 訓練網路模型
history = model.fit(x_train, y_train, 
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))
# 儲存h5權重檔
model.save('MNIST_ep10.h5')


# In[6]:


# 查看參數
model.summary()


# In[7]:


# 畫圖觀察Accuracy與Loss
print(history.history.keys())
f = open('history_key_1.txt', 'w')
#x = str(history.history.keys())
#print(x)
f.write(str(history.history.keys())) 
f.close()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig('[Accuracy]20200805_MNIST_ep10_b128.jpg')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left') 
plt.savefig('[Loss]20200805_MNIST_ep10_b128.jpg')
plt.show()


# In[8]:


# 使用前面讀取的test資料來驗證模型的準確度
# Load the model.
model = load_model('MNIST_ep10.h5')

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)

# Output the result
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




