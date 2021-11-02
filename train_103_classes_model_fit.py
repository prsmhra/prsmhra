#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:00:14 2021

@author: birds
"""

import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
import os
import pickle as pkl
np.random.seed(4)
rn.seed(1234)
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# session_conf = tf.compat.v1.ConfigProtoo(intra_of_parallelism_threads=1,inter_op_parallelism_threads=1)
# from keras import backend as K
# tf.set_random_seed(123)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import heapq
import config_audio
import matplotlib.pyplot as plt
# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=False
sess = tf.compat.v1.Session(config=config)
print(tf.test.gpu_device_name())
import sklearn.metrics as metrics


# Model structure
def DNN_model():
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(585,)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(103, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
    return model


with open("train.pkl", "rb") as f:
    X_train, Y_train = pkl.load(f)
    

with open("val.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)
    
    
    
model = DNN_model()
print(model.summary())
# saving the weights of the traing 
model_file="dnn_best_loss_weights_for_103_species_512_150_epochs.hdf5"
checkpoint = ModelCheckpoint(model_file,monitor = "val_loss", verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]
# fitting the X_train on the model
history = model.fit(X_train,to_categorical(Y_train),epochs=100, batch_size=2000,verbose=0,validation_data=(x_val,to_categorical(y_val)), shuffle=False, callbacks = callback_list)


with open("test.pkl", "rb") as f:
    x_test, y_test = pkl.load(f)
    
Y_pred = np.argmax(model.predict(x_test), axis=-1)
print("Test Accuracy:",round(metrics.accuracy_score(y_test, Y_pred)*100,2),"%")
