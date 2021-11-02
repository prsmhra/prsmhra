#!/usr/bin/env python
# coding: utf-8

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

# softmac function
def softmax(x):
    e_x=np.exp(x-np.max(x))
    return e_x/e_x.sum()

# Model structure
def DNN_model():
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(585,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(103, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    return model


y =[] #empty list
X = np.empty((0,585),dtype=float) # empty list of shape(0,585)
# total_rows = []
# class_name = []

# context for x_train for model
def context_generator(f,y):
    i=0
    files_shape = np.loadtxt(f)
    # files_shape=example.T
    # print(example.shape)
    rows, cols = files_shape.shape
    
    # print(rows,cols)
    context = np.zeros((rows-14,15*cols))
    while i <= (rows-15):
        # print(i)
        ex = files_shape[i:i+15,:].ravel()
        # print("shape 1",ex.shape)
        ex = np.reshape(ex,(1,ex.shape[0]))
        # print("shape 2",ex.shape)
        context[i:i+1,:] = ex
        y.append(config_audio.class_names.index(folder))
        i+=1
    # print("no of window : %d" %i)
    # X = np.append(X,context,axis=0)
    print (context.shape)
    return context

# ---------------------------- Training the Model ----------------------------------------- #

# reading directory one by one
for folder in config_audio.class_names:
    print(folder)
    # context_shape = np.empty((0,585),dtype=float)
    mfcc_file_dir = os.path.join(config_audio.mfcc_train,folder)
    # reading all file in the directory
    for root,dir,file in os.walk(mfcc_file_dir):
        # reading file in the directory one by one
        for f in file:
            print(f)
            mfcc_files = os.path.join(mfcc_file_dir,f)
            fil = np.loadtxt(mfcc_files)
            rows, cols = fil.shape
            if rows > 15:
                # print(mfcc_files)
                # context_generator(mfcc_files)
                context = context_generator(mfcc_files,y) # calling the context_generator function
                # context_shape = np.append(context_shape,context,axis=0)
                X = np.append(X,context,axis=0) # appending the context in x variable
    print(X.shape)
    # class_name.append(folder)
    # total_rows.append(context_shape.shape)

# df= pd.DataFrame()
# df['class_name']=class_name
# df['class_wise_X_train_rows']=total_rows
# df.to_csv("classwise_X_train_rows.csv")
    


# giving the class names labels and one hot vector for training 
clas = []
for i in range(0,len(config_audio.class_names)):
    clas.append(i)  
Y = np.array(y)


# spliting the data for training and testing
X_train, x_val = train_test_split(X, test_size=0.3, random_state=4)
Y_train, y_val = train_test_split(Y, test_size=0.3, random_state=4)

# saving the X_train and Y_train in train.pkl file
with open("train.pkl", "wb") as f:
    pkl.dump([X_train, Y_train], f)
# saving the x_val and y_val in train.pkl file    
with open("val.pkl", "wb") as f:
    pkl.dump([x_val, y_val], f)

# calling the model function
model = DNN_model()
print(model.summary())
# saving the weights of the traing 
model_file="dnn_best_loss_weights_for_104_species.hdf5"
checkpoint = ModelCheckpoint(model_file,monitor = "val_loss", verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]
# fitting the X_train on the model
history = model.fit(X_train,to_categorical(Y_train),epochs=50, batch_size=32,verbose=0,validation_data=(x_val,to_categorical(y_val)), shuffle=False, callbacks = callback_list)


# -------------------------- Testing the Model ---------------------------------- #

file_names =[]
y_test=[]
x_test = np.empty((0,585),dtype=float)
for folder in config_audio.class_names:
    mfcc_files_dir = os.path.join(config_audio.mfcc_test,folder)    
    for r, d, f in os.walk(mfcc_files_dir):
        for filse in f:
            print(filse) 
            input_file = os.path.abspath(mfcc_files_dir+"/"+filse)
            print(input_file) 
            # f = input_file.split('/')
            # file_names.append(f[5]+'/'+f[6]+'/'+f[7]+'/'+f[8])
            mfcc_files = os.path.join(mfcc_files_dir,filse)
            files_shape = np.loadtxt(mfcc_files)
            rows,cols = files_shape.shape
            if rows > 15:
                context = context_generator(mfcc_files,y_test) 
                x_test = np.append(x_test,context,axis=0)
            print(x_test.shape,len(y_test))
with open("test.pkl", "wb") as f:
    pkl.dump([x_test, y_test], f)
# Predicting the classses and accuracy of the model
Y_pred = model.predict_classes(x_test)
print("Test Accuracy:",round(metrics.accuracy_score(y_test, Y_pred)*100,2),"%")

