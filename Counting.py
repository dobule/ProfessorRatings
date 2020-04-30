# Name: Ryan Gelston (rgelston)
# Filename: Counting.py
# Assignment: Term Project
# Description: Contains functions that count and modify tokens in the review
#  text.

#   Structure of proTokenCounts:
#   {'pid' : [Counter(), Counter(), ...],
#      ... 
#   }

import pickle
import numpy as np
from collections import Counter


def num_profs_with_token(profTokenDict):
   """ Returns a Counter with the number of professors a given token appears
         in.
   """
   counter = Counter()
   for profCounts in profTokenDict.values():
      tokensInProf = set()
      for revCtr in profCounts:
         tokensInProf = tokensInProf.union(set(revCtr.keys()))
      counter += Counter(tokensInProf)
   return counter


def num_reviews_with_token(profTokenDict):
   """ Returns a Counter with the number of reviews a given token appears in.
   """
   counter = Counter()
   for profCounts in profTokenDict.values():
      for revCtr in profCounts:
         counter += Counter(revCtr.keys())
   return counter


def num_tokens(profTokenDict):
   """ Returns a Counter with the total times a token appears in all reviews.
   """
   counter = Counter() 
   for profCounts in profTokenDict.values():
      for revCtr in profCounts:
         counter += revCtr
   return counter


def create_prof_token_dict(profs, token_f):
   """ Returns a dictionary of counter objects for each professor. The key to
       the first dictionary is the professor's pid. The keys for each 
       counter object is the token with the value being the occurance of
       said token.

       profs -- Professor dictionary as found in profDicts.pkl
       token_f -- Function that returns tokens to be considered when given
                  the text of a review, as found in profDicts
   """
   profsTupleDict = {}
   for prof in profs:
      profsTupleDict[prof['pid']] = prof_rev_counters(prof, token_f)
   return profsTupleDict


def prof_rev_counters(prof, token_f):
   """ Returns a list of counter objects that hold the token count of each
       review.
   """
   countList = []
   for rev in prof['reviews']:
      countList.append(count_rev(rev, token_f))
   return countList


def count_rev(review, token_f):
   """ Returns counter object for the tokens in the reviews text """
   return Counter(token_f(review['text']))


def combine_rev_counters(counters):
   """ Returns a dictionary that holds the aggregate count for each counter
       in counters.
   """
   aggregate = Counter()
   for counter in counters:
      aggregate += counter
   return aggregate


def create_single_tokens(text, stopwords=None, stmr=None):
   """ Stemms and removes stopwords from the list of words 'text' """
   tokens = []
   for token in text:
      if type(stmr) != type(None):
         token = stmr.stem(token)
      if type(stopwords) == type(None) or token not in stopwords:
         tokens.append(token)
   return tokens
         

def create_tuple_tokens(text, stopwords=None, stmr=None):
   """ Creates tuples from the words in text. Tuples with stopwords in them
       are not included.
   """
   tokens = []
   lastToken = None
   for token in text:
      if type(stmr) != type(None):
         token = stmr.stem(token)
      if type(stopwords) == type(None) or token not in stopwords:
         if lastToken != None:
            tokens.append(lastToken + '-' + token)
         lastToken = token
      else:
         lastToken = None
   return tokens


def create_single_tuple_tokens(text, stopwords=None, stmr=None):
   """ Creates an array of tokens from text. Tokens include stemmed words and
       tuples of consecutive stemmed words that are not in the stopwords.

       text -- a list of tokenized strings 
       stopwords -- a set of words to not include
       stmr -- stemmer object
   """
   
   if text == []:
      return []

   tokens = []
   lastToken = None

   for token in text:
      if type(stmr) != type(None):
         token = stmr.stem(token)
      if type(stopwords) == type(None) or token not in stopwords:
         tokens.append(token)
         if lastToken != None:
            tokens.append(lastToken + '-' + token)
         lastToken = token
      else:
         lastToken = None 
   return tokens


