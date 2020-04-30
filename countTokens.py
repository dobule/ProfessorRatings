# Name: Ryan Gelston (rgelston)
# Filename: countTokens.py
# Assignment: Term Project
# Description: CMD line utility for creating tokens from reviews 

import sys
import os
import pickle
import ReadFromFile as read
import WriteToFile as write
import FileNames as fn
import Counting as count
import PlotData as plot
from nltk.stem.lancaster import LancasterStemmer


def print_usage_message():
   print("python3 countTokens.py [-ss] [-tup | -stup] [-save] [-h]")
   print("\tDEFUALT: Count singletons without stemming or removing stopwords")
   print("\t-ss -- Stem words and remove stopwords")
   print("\t-tup -- Count tuples as tokens")
   print("\t-stup -- Counts singletons and tuples as tokens")
   print("\t-save -- Saves plot to image in ./figure/tokenCount/")
   print("\t-h -- Print usage message")


def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   stmr = None
   stopwords = None
   if '-ss' in sys.argv:
      stmr = LancasterStemmer()
      stopwords = read.stopwords(stmr)

   countNames = fn.create_token_count_names(sys.argv)
   rawTokenCountName = countNames[0]
   revTokenCountName = countNames[1]
   profTokenCountName = countNames[2]

   rawTokens = read.token_count(rawTokenCountName, True)
   revTokens = read.token_count(revTokenCountName, True)
   profTokens = read.token_count(profTokenCountName, True)

   if rawTokens == None or revTokens == None or profTokens == None:
      profTokenDict = grab_prof_token_dict(stopwords, stmr)

      if rawTokens == None:
         rawTokens = grab_token_count(profTokenDict,
                                      count.num_tokens,
                                      rawTokenCountName)
      if revTokens == None:
         revTokens = grab_token_count(profTokenDict,
                                      count.num_reviews_with_token,
                                      revTokenCountName)

      if profTokens == None:
         profTokens = grab_token_count(profTokenDict,
                                       count.num_profs_with_token,
                                       profTokenCountName)

   
   plotName = create_plot_name()
   plotFileName = None
   if '-save' in sys.argv:
      plotFileName = fn.create_count_plot_name(sys.argv)
     
   plot.token_counts(rawTokens, revTokens, profTokens, 
      plotFileName, plotName)


def grab_prof_token_dict(stopwords, stmr):

   filename = fn.create_prof_token_dict_name(sys.argv)

   if os.path.exists(filename):
      with open(filename, 'rb') as f:
         profTokenDict = pickle.load(f)
      return profTokenDict

   token_f = lambda t: count.create_single_tokens(t, stopwords, stmr)
   if '-tup' in sys.argv:
      token_f = lambda t: count.create_tuple_tokens(t, stopwords, stmr)
   elif '-stup' in sys.argv:
      token_f = (lambda t: 
                  count.create_single_tuple_tokens(t, stopwords, stmr))

   profs = read.prof_dicts()

   profTokenDict = count.create_prof_token_dict(profs, token_f)

   with open(filename, 'wb') as f:
      pickle.dump(profTokenDict, f)

   return profTokenDict


def grab_token_count(profTokenDict, token_counter_f, fileName):
   tokenCount = token_counter_f(profTokenDict)
   tokenCount = [(k, v) for k, v in tokenCount.items()]
   write.token_count(tokenCount, fileName)
   return tokenCount
  

def create_plot_name():
   plotName = "Token Count: "
   if '-tup' in sys.argv:
      plotName += "Tuples"
   elif '-stup' in sys.argv:
      plotName += "Singletons and Tuples"
   else:
      plotName += "Singletons"

   if '-ss' in sys.argv:
      plotName += ", Stemmed and Removed Stopwords"

   return plotName


if __name__=="__main__":
   main()


