# Name: Ryan Gelston (rgelston)
# Filename: findCorrelations.py
# Assignment: Term Project
# Description: Finds correlations between tokens in the token vector

import sys
import os
import Stats as stat
import FileNames as fn
import ReadFromFile as read
import WriteToFile as write
import PlotData as plot

def print_usage_message():
   print("python3 findCorrelations.py [-h]")
   print("\t-ss -- Use stemmed tokens with stopwords removed")
   print("\t-tup -- Use tuples as tokens")
   print("\t-stup -- Use tuples and singletons as tokens")
   print("\t-minCount <int> -- Only include tokens that appear in "
      + "int reviews.")
   print("\t-save -- Save the plot to figures/correlations/")
   print("\t-h -- Print usage message")


def main():
  
   if '-h' in sys.argv:
      print_usage_message()
      exit()

   vectorFileName = fn.create_prof_vect_name(sys.argv)
   corrFileName = fn.create_correlations_name(sys.argv)

   if not os.path.exists(corrFileName):
      tokenVects = read.word_vects(vectorFileName)
      if tokenVects is None:
         print("Specified vector file not found.")
         print("To create vectors use 'createProfVectors.py'")
         exit()
      ratingVect = read.overall_rating_vect(vectorFileName)
      vocabVect = read.vocab_from_vect_file(vectorFileName)
      corrTups = stat.find_correlations(tokenVects, ratingVect, vocabVect)
      write.token_correlations(corrTups, corrFileName)
   else: 
      corrTups = read.token_correlations(corrFileName)

   corrPlotFileName = None
   if '-save' in sys.argv:
      corrPlotFileName = fn.create_correlations_plot_name(sys.argv)

   # Plot correlations
   plot.tuple_pair_score_correlation(corrTups,
      title=plot.create_token_pair_score_correlation_name(sys.argv),
      saveFile=corrPlotFileName) 

if __name__=="__main__":
   main()
