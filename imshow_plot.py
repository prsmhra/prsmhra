#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:57:52 2021

@author: birds
"""


import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
import librosa as lb
import glob
import python_speech_features
import os
import pickle as pkl
np.random.seed(4)
rn.seed(1234)
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import heapq
# import config_audio
import matplotlib.pyplot as plt
# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
# print(tf.test.gpu_device_name())
import sklearn.metrics as metrics


def extract_delta_features(feats):
    feats1 = lb.feature.delta(feats)
    feats1 = np.asarray(feats1)
    return feats1



def softmax(x):
    e_x=np.exp(x-np.max(x))
    return e_x/e_x.sum()



# Model structure
def DNN_model():
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(585,)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(103, activation='softmax'))
    print("Loaded model from disk")
    model.load_weights("dnn_best_loss_weights_for_103_species_512.hdf5")
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    return model

def context_generator(f):
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
        # y.append(config_audio.class_names.index(folder))
        i+=1
    # print("no of window : %d" %i)
    # X = np.append(X,context,axis=0)
    # print(context.shape)
    return context

model = DNN_model()
f = sorted( filter( os.path.isfile, glob.glob('/home/birds/Documents/swift_audio_mfcc/SWIFT02_20210818_070001' + '/*') ) )
predictions = []
for ff in f:
    print(ff)
    # mfcc_files = os.path.join(file,filse)
    # print(mfcc_files)
    # mfcc_file = egy_based_mfcc(ff)
    files_shape = np.loadtxt(ff)
    rows,cols = files_shape.shape
    if rows > 15:
        context = context_generator(ff) 
        # x_test = np.append(x_test,context,axis=0)
        # print(x_test.shape)
        # print(context.shape)
        Y_pred = model.predict(context)
        sum1=np.sum(Y_pred, axis=0)
        a=np.asarray(sum1)
        labels=heapq.nlargest(5, range(len(a)), a.take)
        label= np.argmax(np.asarray(sum1))
        # print(label)
        confidence_matrix = softmax(sum1)
        confidence_matrix = confidence_matrix*100
        predictions.append(confidence_matrix)
        # print(confidence_matrix)
pred_folder = "/home/birds/Documents/save_predictions/"+ff.split('/')[-1][:-4].split('-')[0]
pred_file = ff.split('/')[-1][:-4].split('-')[0]
try:
    os.makedirs(pred_folder,0o775)
except FileExistsError:
    print('already exists')
pred_vals = np.asarray(predictions).transpose()
np.savetxt(pred_folder+'/'+pred_file+".txt",predictions,fmt='%f')


fig, ax = plt.subplots(figsize=(40,40))
ax.set_title(pred_file, fontsize=30)
ax.set_yticks(np.arange(len(config_audio.class_names)))
ax.set_yticklabels(config_audio.class_names,fontsize=20)

plt.xticks(fontsize=20)

img = ax.imshow(pred_vals,aspect='auto')
fig.savefig(pred_folder+'/'+pred_file+'.png')


