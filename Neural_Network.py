import os
import sys

import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from keras import layers 
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train =  x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
print(x_train.shape, x_test.shape)

# model = keras.Sequential(
#     [
#         keras.Input(shape=(28*28)),
#         layers.Dense(units = 512, activation = 'relu'),
#         layers.Dense(units = 256, activation = 'relu'),      
#     ]
# )

inputs = keras.Input(shape=(28*28))
x =  Dense(units = 512, activation = 'relu')(inputs)
x =  Dense(units = 256, activation = 'relu')(x)
outputs = Dense(units = 10, activation = 'softmax')(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=  False),
    optimizer = Adam(learning_rate = 0.001),
    metrics = ['accuracy'],
)

model.fit(x_train,y_train,batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(x_test,y_test,batch_size = 32, verbose = 2)





