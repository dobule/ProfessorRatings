# Name: Ryan Gelston (rgelston)
# Filename: runFFNN.py
# Assignment: Term Project
# Description: Commandline utility for training a feed-forward neural-network
#  using professor vectors.

import sys
import numpy as np
import PlotData as plot
import FileNames as fn
import ReadFromFile as read
import WriteToFile as write
import FfnnAlgorithm as ffnn
import VectorProcessing as vp

from sklearn.model_selection import train_test_split


def print_usage_message():
   print("python3 runFFNN.py [-ss] [-tup | -stup] [-minCount <num>]")
   print("\t-ss -- Uses stemmed and stopped vectors")
   print("\t-tup -- Use vectors with tuple tokens")
   print("\t-stup -- Use vectors with both singleton and tuple tokens")
   print("\t-minCount <num> -- Use vectors with tokens that appear in "
      + "at least num reviews")
   print("\t-corr <minCount> <minScore> -- Include token pairs with at least"
      + " minCount occurances in prof vectors and a minimum absolute" 
      + " correlation score of minScore")
   print("\t-deep -- Use deep model")
   print("\t-h -- print usage message")


def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   vectorFileName = fn.create_prof_vect_name(sys.argv, True)
   tokenVects = read.word_vects(vectorFileName)
   if tokenVects is None:
      print("Could not find token vects")
      print("Use 'createProfVectors.py' to create vectors")
      exit()

   tokenVects = vp.process_token_vectors(tokenVects, sys.argv)

   if '-d' in sys.argv:
      ratings = read.difficulty_rating_vect(vectorFileName)
   else:
      ratings = read.overall_rating_vect(vectorFileName)


   # Create Training and validation sets
   pidVect = read.pid_vect(vectorFileName)

   nonSingleSmallIdxs = ffnn.non_single_small_idxs(pidVect)
   singleIdxs = vp.pids_to_idxs(pidVect,
                  read.pids_file(fn.PidsSingleRevFile))
   smallIdxs = vp.pids_to_idxs(pidVect,
                  read.pids_file(fn.PidsSmallRevLenFile))
   singleSmallIdxs = list(set(singleIdxs).union(set(smallIdxs)))
   singleSmallIdxs.sort()
   singleSmallIdxs = np.array(singleSmallIdxs)

   trainingVects = tokenVects[nonSingleSmallIdxs, :]
   trainingRatings = ratings[nonSingleSmallIdxs]

   validVects = tokenVects[singleSmallIdxs, :]
   validRatings = ratings[singleSmallIdxs]

   print(trainingVects.shape, trainingRatings.shape, 
         validVects.shape, validRatings.shape)
   """
  
   xTrain, xValid, yTrain, yValid = train_test_split(tokenVects, ratings,
                                                      test_size=0.3)
   """ 
   # Select and train model
   if '-deep' in sys.argv:
      model = ffnn.deep_model(tokenVects.shape[1])
   else:
      model = ffnn.shallow_model(tokenVects.shape[1])

   history = model.fit(trainingVects, trainingRatings,
                       epochs=10,
                       batch_size=5,
                       validation_data=(validVects, validRatings))

   plotTitle = plot.ffnn_error_title(sys.argv)
   outfile = None
   if '-save' in sys.argv:
      outfile = fn.create_ffnn_plot_name(sys.argv)

   plot.ffnn_error(history, title=plotTitle, filename=outfile)



if __name__=="__main__":
   main()


