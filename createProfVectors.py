# Name: Ryan Gelston (rgelston)
# Filename: createProfVectors.py
# Assignment: Term Project
# Description: Creates vector files 

import sys
import Counting as count
import ReadFromFile as read
import WriteToFile as write
import FileNames as fn
import VectorProcessing as vp
import Counting as count
import Stats as stat
from nltk.stem.lancaster import LancasterStemmer

def print_usage_message():
   print("python3 createVectors.py [-ss] [-tup | -stup] [-minCount <int>] "
          + "[-corr <minCnt> <minScore>] [-h]")
   print("\t-ss -- Stem words and remove stopwords")
   print("\t-tup -- Count tuples as tokens")
   print("\t-stup -- Counts singletons and tuples as tokens")
   print("\t-minCount <int> -- Only include tokens that appear in at least "
            + "<int> reviews")
   print("\t-corr <minCnt> <minScore> -- Include token pairs with at least "
            + "minCnt occurances in prof vectors and a min absolute "
            + "correlation score of minScore")
   print("\t-h -- Print usage message")


def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   tokenSchema = vp.create_token_schema(sys.argv)
   profVects, pidsNotIncl = vp.create_prof_vectors(tokenSchema, sys.argv)
   profVectFileName = fn.create_prof_vect_name(sys.argv, True)
   write.prof_vects(profVects, pidsNotIncl, tokenSchema, profVectFileName)


if __name__=="__main__":
   main()
