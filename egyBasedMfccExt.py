#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:15:20 2021

@author: birds
"""

import numpy as np
import matplotlib.pyplot as plt
import python_speech_features
import librosa as lb
import config_audio
import os

# Delta Features Generations
def extract_delta_features(feats):
    feats1 = lb.feature.delta(feats)
    feats1 = np.asarray(feats1)
    return feats1


# filePath = 'train_test_data/train/alexandrine_parakeet/alexandrine_parakeet_12.wav'


def wavToMfcc(src, dest):
    for clas in config_audio.class_names: # reading directory 
#        print(clas)
        wav_dir = os.path.join(src,clas)
        # print(wav_dir)
        mfcc_dir = os.path.join(dest, clas)
        # print(mfcc_dir)
        try:
            os.makedirs(mfcc_dir,0o775)
        except FileExistsError:
            print("Already Exists!!")
        for root, dir, file in os.walk(wav_dir): # reading all files in a directory
            for f in file: # reading single file from a directory
                # print(f)
                filePath = os.path.join(wav_dir,f)
                print(filePath)

                # to retain the native sampling rate
                y, sr = lb.load(filePath,sr=None)
                # frate = 100
                winSize = int(np.ceil(0.02*sr)) # in samples
                hopLength = int(np.ceil(0.01*sr))
                
                # frame the signal
                sigFrames = lb.util.frame(y,frame_length=winSize,hop_length=hopLength)
                print(sigFrames.shape)
                # compute energy
                sigSTE = np.sum(np.square(sigFrames),axis=0)
                # plt.plot(sigSTE)
                
                meanEgy = np.mean(sigSTE)
                # get logical indices
                x = sigSTE>meanEgy
                # speech egy values
                speechEgy = sigSTE[x]
                # nonspeech egy vales
                nonSpeechEgy = sigSTE[~x]
                
                # indices where egy greater than threshold
                speechIndices = np.where(sigSTE>meanEgy)
                
                mfccs=python_speech_features.base.mfcc(y, sr, winlen=0.02, winstep=0.01, numcep=13, nfft=1024, nfilt=32, appendEnergy=True)
                feats1 = extract_delta_features(mfccs)
                feats2 = extract_delta_features(feats1)
                initial = np.concatenate((mfccs,feats1), axis=1)
                final_39_dim = np.concatenate((initial,feats2), axis=1)
                final = final_39_dim-np.min(final_39_dim)
                nfinal = final/float(np.max(final))
                # plt.plot(nfinal)
                egyMfccFinal = nfinal[speechIndices[0]]
                print(egyMfccFinal.shape)
                mfcc = open(mfcc_dir+'/'+f[:-4]+'.mfcc','w')
                np.savetxt(mfcc,egyMfccFinal,fmt='%f')

wavToMfcc(config_audio.train_pth,config_audio.mfcc_train)
# wavToMfcc(config_audio.test_pth,config_audio.mfcc_test)