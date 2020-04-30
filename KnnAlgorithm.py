# Name: Ryan Gelston
# File: KnnAlgorithm.py
# Assignment: Term Project
# Description: Contains the knn algorithm along with functions for 
#  analyzing the results of a knn run.

import numpy as np

NO_SIM_VAL = -1

def knn(predIdx, scores, maxK, simMat, weighted=True):
   """ Runs the KNN algorithm on a single index in the data array 

       Parameters:
         predIdx -- The index of the vector in data to predict
         scores -- value of each data vector
         maxK -- Maximum k value to compute for each data point
         simMat -- a 2D array with cached similarity values

       Returns: A numpy array of length maxK with the prediction with k-value
         n being at index n-1.
   """

   predictions = np.zeros(maxK, dtype=float)
   simScorePairs = []

   # Find the similarity for the predicted vector and all other vectors
   for idx in range(scores.shape[0]):
      # Don't include the vector being predicted
      if idx == predIdx:
         continue

      # Similarity functions are symmetrical, so we only need to store and
      #  retrieve the values for pairs of indeies (i, j) where i < j
      if idx < predIdx:
         idxTup = (idx, predIdx)
      else:
         idxTup = (predIdx, idx)

      # Grab the similarity value
      simScorePairs.append((simMat[idxTup], scores[idx]))

   simScorePairs.sort(reverse=True)
   numSum = 0.0 # Numerator of weighted sum for prediction
   denSum = 0.0 # Denominator of weighted sum for prediction

   # Calculate predictions for different k values
   for idx in range(maxK):
      curTup = simScorePairs[idx]
      if weighted:
         numSum += curTup[0] * curTup[1]
         denSum += curTup[0]
      else:
         numSum += curTup[1]
         denSum += 1

      predictions[idx] = numSum / denSum

   return predictions


def knn_dataset(scores, maxK, simMat, weighted=True, idxs=None):
   
   # Stores computed similarity values
   predictions = np.zeros((scores.shape[0], maxK), dtype=float)

   if idxs == None:
      idxs = range(scores.shape[0])

   for idx in idxs:
      predictions[idx] = knn(idx, scores, maxK, simMat, weighted)

   return predictions


def get_similarity_matrix(data, sim_f):
   
   simMat = np.full((data.shape[0], data.shape[0]),
                     NO_SIM_VAL,
                     dtype=float)

   for idx1 in range(data.shape[0]):
      for idx2 in range(idx1+1, data.shape[0]):
         simMat[idx1, idx2] = sim_f(data[idx1], data[idx2])

   return simMat


  
