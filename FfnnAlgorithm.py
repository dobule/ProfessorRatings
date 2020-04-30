# Name: Ryan Gelston (rgelston)
# Filename: FfnnAlgorithm.py
# Assignment: Term Project
# Description: Contains different FFNN architectures

import ReadFromFile as read
import FileNames as fn

import numpy as np

from keras.optimizers import SGD, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout

def shallow_model(inputLen):

   model = Sequential([
      Dense(int(0.5 * inputLen), activation='relu', input_shape=(inputLen,)),
      Dense(1)])

   model.compile(optimizer=RMSprop(),
                 loss='mean_absolute_error',
                 metrics=['mae'])

   return model


def deep_model(inputLen):

   model = Sequential([
      Dense(int(1.5 * inputLen), activation='relu', input_shape=(inputLen,)),
      Dense(int(inputLen), activation='relu'),
      Dense(int(0.75 * inputLen), activation='relu'),
      Dense(int(0.5 * inputLen), activation='relu'),
      Dense(1)])

   model.compile(optimizer=RMSprop(),
                 loss='mean_absolute_error',
                 metrics=['mae'])

   return model




"""
def deep_model(inputLen):
   return Sequential([
      Dense( , activation= , input_shape=(inputLen,)),
      Dense( , activation= )])
"""

def non_single_small_idxs(pidVect):
   singlePids = set(read.pids_file(fn.PidsSingleRevFile))
   smallPids = set(read.pids_file(fn.PidsSmallRevLenFile))
   nonSingleSmallIdxs = [idx for idx, pid in enumerate(pidVect)
                           if pid not in singlePids
                              and pid not in smallPids]
   return np.array(nonSingleSmallIdxs)

