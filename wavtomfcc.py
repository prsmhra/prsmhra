# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import librosa as lb
import os
import sys
import python_speech_features
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import config_audio

def extract_delta_features(feats):
    feats1 = lb.feature.delta(feats)
    feats1 = np.asarray(feats1)
    return feats1


def wavToMfcc(src, dest):
    for clas in config_audio.class_names:
#        print(clas)
        wav_dir = os.path.join(src,clas)
        # print(wav_dir)
        mfcc_dir = os.path.join(dest, clas)
        # print(mfcc_dir)
        try:
            os.makedirs(mfcc_dir,0o775)
        except FileExistsError:
            print("Already Exists!!")
        for root, dir, file in os.walk(wav_dir):
            for f in file:
                # print(f)
                wav_file = os.path.join(wav_dir,f)
                print(wav_file)
                # y, sr = librosa.load(wav_file)
                # print(sr)
                # mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=39, hop_length=1024)
                # sr1,y1=wav.read(wav_file)
                y, sr = lb.load(wav_file, sr=None)
                # print(sr1,sr)
                mfccs=python_speech_features.base.mfcc(y, sr, winlen=0.02, winstep=0.01, numcep=13, nfft=1024, nfilt=32, appendEnergy=True)
                #mfcc1=librosa.feature.mfcc(y, sr, 5, winstep=0.01, n_mfcc=12, nfilt=26, nfft=441, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
                #mfccs = librosa.feature.mfcc(y=y, sr=sr,n_fft=4096, hop_length = 441,n_mfcc=13,n_mels=32,norm=1)
                #mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)
                #Tfeats = mfccs.transpose()
                print(mfccs.shape)
                feats1 = extract_delta_features(mfccs)
                Tfeats1 = feats1.transpose()
                feats2 = extract_delta_features(feats1)
                initial = np.concatenate((mfccs,feats1), axis=1)
                # print(initial.shape)
                final_39_dim = np.concatenate((initial,feats2), axis=1)
                final = final_39_dim-np.min(final_39_dim)
                nfinal = final/float(np.max(final))
                mfcc = open(mfcc_dir+'/'+f[:-4]+'.mfcc','w')
                np.savetxt(mfcc,nfinal,fmt='%f')

# wavToMfcc(config_audio.train_pth,config_audio.mfcc_train)
wavToMfcc(config_audio.test_pth,config_audio.mfcc_test)