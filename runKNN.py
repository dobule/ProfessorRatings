# Name: Ryan Gelston
# Assignment: Term Project
# Description: Runs KNN on the dataset, looking at the accuracy for multiple
#  k values up to a specified max k value.


import sys
import os
import numpy as np
import VectorProcessing as vp
import KnnAlgorithm as knn
import FileNames as fn
import ReadFromFile as read
import WriteToFile as write
import PlotData as plot

MaxK = 250

def print_usage_message():
   print("python3 runKNN.py [-d] [-ss] [-tup | -stup] "
          + "[-minCount <int>] [-tf | -tfidf] [-cos | -pear] [-unweighted] "
          + "[-corr <minCnt> <minScore>] [-maxK <int>] [-h]")
   print("\t-d -- Use difficulty rating, rather than overall")
   print("\t-ss -- Use semmed vectors with stopwords removed")
   print("\t-tup -- Use vectors that only count tuples")
   print("\t-stup -- Use vectors that count singletons and tuples")
   print("\t-minCount <int> -- Use vectos that only include tokens that "
            + "appear in at least <int> reviews")
   print("\t-tf -- Use term frequency vector")
   print("\t-tfidf -- Use tf-idf vector")
   print("\t-cos -- Use cosine similarity")
   print("\t-pear -- Use absolute pearson correlation as a similarity metric")
   print("\t-unweighted -- Do not weight ratings with similarity metric "
          + "when predicting score")
   print("\t-corr <minCnt> <minScore> -- Include token pairs with at least "
            + "minCnt occurances in prof vectors and a min absolute "
            + "correlation score of minScore")
   print("\t-maxK <int> -- Only plot k values under int")
   print("\t-h -- Print usage message")


def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   vectFileName = fn.create_prof_vect_name(sys.argv, True)
   simMatFileName = fn.create_sim_mat_name(sys.argv)
   predsFileName = fn.create_preds_name(sys.argv)

   print(vectFileName)
   print(simMatFileName)
   print(predsFileName)

   # Grab the ratings vector 
   if '-d' in sys.argv:
      ratings = read.difficulty_rating_vect(vectFileName)
   else:
      ratings = read.overall_rating_vect(vectFileName)

   # Assign similarity metric
   sim_f = vp.inverse_euclidean_distance
   if '-cos' in sys.argv:
      sim_f = vp.cosine_similarity
   elif '-pear' in sys.argv:
      sim_f = vp.abs_pearson_correlation

   # Set if weighted or not
   weighted = True
   if '-unweighted' in sys.argv:
      weighted = False

   # Grab predictions or create them if not available
   predictions = read.knn_predictions(predsFileName)
   if predictions is None:

      simMat = read.similarity_matrix(simMatFileName)
      if simMat is None:
         wordVects = read.word_vects(vectFileName)
         if wordVects is None:
            print("Vector file " + vectFileName + " does not exist")
            exit()
         wordVects = vp.process_token_vectors(wordVects, sys.argv)
         simMat = knn.get_similarity_matrix(wordVects, sim_f)
         write.similarity_matrix(simMat, simMatFileName)

      predictions = knn.knn_dataset(ratings,
                                    MaxK,
                                    simMat,
                                    weighted)
      write.knn_predictions(predictions, predsFileName) 
      

   idxToPlot = None

   if '-maxK' in sys.argv:
      maxK = int(sys.argv[sys.argv.index('-maxK') + 1])
      predictions = predictions[:,:maxK]

   pidVect = read.pid_vect(vectFileName)
   singleRevIdxs = vp.pids_to_idxs(
                    pidVect, read.pids_file(fn.PidsSingleRevFile))
   smallLenIdxs = vp.pids_to_idxs(
                    pidVect, read.pids_file(fn.PidsSmallRevLenFile))

   plotFileName = None
   if '-save' in sys.argv:
      plotFileName = fn.create_knn_accuracy_plot_name(sys.argv)

   # Output results of the run
   plot.knn_error(predictions, ratings, 
                  title=plot.create_knn_error_title(sys.argv),
                  idxToPlot=[singleRevIdxs, smallLenIdxs],
                  subTitles=["Error with profs with one review",
                             "Error with profs with aggrigate review "
                              + "lengths one std div above the mean "
                              + "review length or less"],
                  saveFile=plotFileName)

if __name__=="__main__":
   main()
