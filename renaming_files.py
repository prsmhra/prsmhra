#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:29:22 2021

@author: birds
"""

import glob
import shutil
import os


list_of_files = sorted( filter( os.path.isfile, glob.glob('/home/birds/Documents/swift_audio/' + '*') ) )

for file_path in list_of_files:
    # print(file_path)
    fl = file_path.split('/')[-1]
    # print(fl)
    f = fl.split('-')[0]
    print(f)
    dest_dir = '/home/birds/Documents/swift_audio/'+f
    try: 
        os.makedirs(dest_dir,0o775)
    except FileExistsError:
        print("already exists")
    destination = dest_dir+'/'+fl
    try:
        shutil.copy(file_path, destination)
        print("File copied successfully.")
 
    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    # f = fl.split('-')[1][:-4]
    # # print(len(f))
    # if len(f)==1:
    #     # print(f)
    #     f = '00'+f
    #     # print(f)
    #     fle = file_path.split('-')[0]
    #     nfl = fle+'-'+f+'.wav'
    #     print(nfl)
    #     os.rename(file_path,nfl)
    # if len(f)==2:
    #     # print(f)
    #     f = '0'+f
    #     # print(f)
    #     fle = file_path.split('-')[0]
    #     nfl = fle+'-'+f+'.wav'
    #     print(nfl)
    #     os.rename(file_path,nfl)
    
    
    