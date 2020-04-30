# Name: Ryan Gelston
# File: VectorProcessing.py
# Assignment: Term Project
# Description: Contains functions for converting raw wordcount vectors into 
#  term-frequency vectors and term-frequency inverse-document-frequency 
#  vectors.

import pickle
import numpy as np
import Counting as count
import FileNames as fn
import ReadFromFile as read
from nltk.stem.lancaster import LancasterStemmer


###########################################################
# Token Schema creation functions
###########################################################

def create_token_schema(argv):
   if '-corr' in argv:
      return token_schema_from_correlations(argv)
   return token_schema_from_count(argv)


def token_schema_from_correlations(argv):
   corrFileName = fn.create_correlations_name(argv)
   corrTups = read.token_correlations(corrFileName)
   if corrTups is None:
      print("Correlations file not found")
      print("Create correlations file with 'findCorrelations.py'")
      exit()

   corIdx = argv.index('-corr')
   minCount = int(argv[corIdx + 1])
   minScore = float(argv[corIdx + 2])

   reducedTups = [(cor[0], cor[1]) for cor in corrTups
                   if cor[2] >= minCount and cor[3] >= abs(minScore)]

   tokenSet = set()
   for tok1, tok2 in reducedTups:
      tokenSet.add(tok1)
      tokenSet.add(tok2)

   tokenSchema = list(tokenSet)
   tokenSchema.sort()
   return tokenSchema


def token_schema_from_count(argv):
   countsFileName = fn.create_token_count_names(argv)
   countsFileName = countsFileName[1] # Num revs token appears in 
   tokenCounts = read.token_count(countsFileName)
   if tokenCounts is None:
      print("Token count file not found.")
      print("Create token count file using 'countTokens.py'")
      exit()

   minCount = int(argv[argv.index('-minCount') + 1])
   tokenSchema = [t for t, c in tokenCounts.items()
                    if c >= minCount]
   tokenSchema.sort()
   return tokenSchema


###########################################################
# Vector creation functions
###########################################################

def create_prof_vectors(
      tokenSchema, argv, profDicts=None, profTokenDict=None):
   """ Create token count vectors for the aggrigate reviews of each 
       professor.
   """

   if profDicts is None:
      profDicts = read.prof_dicts()
   
   if profTokenDict is None:
      ptdName = fn.create_prof_token_dict_name(argv)
      profTokenDict = read.prof_token_dicts(ptdName)

   schemaDict = value_idx_dict(tokenSchema)
  
   profVects = []
   pidsNotIncluded = []
   for prof in profDicts:
      newVect = create_prof_vector(
                  prof, 
                  count.combine_rev_counters(
                     profTokenDict[prof['pid']]),
                  schemaDict)
      if newVect['token_vect'] is None:
         pidsNotIncluded.append(newVect['pid'])
      else:
         profVects.append(newVect)
        
   pidsNotIncluded.sort()

   return profVects, pidsNotIncluded


def create_prof_vector(prof, tokenCounter, schemaDict):   
   vect = {}
   vect['pid'] = prof['pid']
   vect['rating_difficulty'] = prof['rating_difficulty']
   vect['rating_overall'] = prof['rating_overall']
   vect['token_vect'] = create_token_vect(
                           tokenCounter,
                           schemaDict)
   return vect


def create_rev_vectors(
      tokenSchema, argv, profDicts=None, profTokenDict=None):

   if profDicts is None:
      profDicts = read.prof_dicts()
   
   if profTokenDict is None:
      ptdName = fn.create_prof_token_dict_name(argv)
      profTokenDict = read.prof_token_dicts(ptdName)

   schemaDict = value_idx_dict(tokenSchema)
   
   revVects = []
   for prof in profDicts:
      for rev in prof['reviews']:
         revVects.append(
            create_rev_vector(rev,
                              schemaDict))


def create_rev_vector(rev, schemaDict):
   #TODO: Impliment
   pass

def create_token_vect(tokenCounter, schemaDict):
   """ Returns None if zero vector """
   vect = np.zeros(len(schemaDict), dtype=int)
   setValue = False
   for tok, count in tokenCounter.items():
      if tok in schemaDict.keys():
         setValue = True
         vect[schemaDict[tok]] = count
   if not setValue:
      return None
   return vect


def value_idx_dict(aList):
   aDict = {}
   for idx, val in enumerate(aList):
      aDict[val] = idx
   return aDict



###########################################################
# Modifies word count vectors. To be done after vector
# creation.
###########################################################

def to_tf_vect(vect): 
   totalWords = np.sum(vect)
   return vect / totalWords


def to_tf_idf_vect(vect, idf_vect):
   tf_vect = to_tf_vect(vect)
   return tf_vect * idf_vect


def create_idf_vect(vocab, numProfs, argv):
   """ vocab is expected to be a python list """

   countFileName = fn.create_token_count_names(argv)
   countFileName = countFileName[2]

   tokCounts = read.token_count(countFileName)

   countVect = np.zeros(len(vocab), dtype=float)

   for idx, word in enumerate(vocab):
      countVect[idx] = tokCounts[word]

   return np.log(numProfs / countVect)


def process_token_vectors(vects, argv):
   if '-tf' in argv:
      vects = np.apply_along_axis(to_tf_vect, 1, vects)
   elif '-tfidf' in argv:
      vocab = read.vocab_from_vect_file(
                  fn.create_prof_vect_name(argv))
      idfVect = create_idf_vect(vocab, vects.shape[0], argv)
      print(idfVect.shape, vects.shape)
      vects = np.apply_along_axis(
                  lambda x: to_tf_idf_vect(x, idfVect),
                  1,
                  vects)
   return vects
      

###########################################################
# Similarity metrics
###########################################################

def return_one(vec1, vec2):
   """ Every vector pair is equally weighted """
   return 1

def inverse_euclidean_distance(vec1, vec2):
   """ Returns euclidean distance of two vectors """
   return 1 / np.sqrt(
                np.sum(
                  np.square(vec1 - vec2)))


def cosine_similarity(vec1, vec2):
   """ Returns cosine similarity of two vectors """
   numerator = np.sum(vec1 * vec2)
   denominator = np.sqrt(
                   np.sum(np.square(vec1))
                   * np.sum(np.square(vec2)))

   return numerator / denominator

def abs_pearson_correlation(vec1, vec2):
   """ Returns the pearson correlation of two vectors """
   redVec1 = []
   redVec2 = []
   # Reduce the input vectors to entries where both are non-zero
   for c1, c2 in zip(vec1, vec2):
      if c1 != 0 and c2 != 0:
         redVec1.append(c1)
         redVec2.append(c2)

   # Return 0 if there are no non-zero values
   if len(redVec1) < 1:
      return 0

   redVec1 = np.array(redVec1, dtype=float)
   redVec2 = np.array(redVec2, dtype=float)
   meanV1 = np.nanmean(redVec1)
   meanV2 = np.nanmean(redVec2)
   adjV1 = redVec1 - meanV1
   adjV2 = redVec2 - meanV2
   numerator = np.sum(adjV1 * adjV2)
   denominator = ((np.sum(adjV1 ** 2) * np.sum(adjV2 ** 2)) ** 0.5)

   # Avoid divide by zero error
   if denominator == 0:
      return 0

   return np.absolute(numerator / denominator)


###########################################################
# Utility functions 
###########################################################

def pids_to_idxs(pidVect, pids):
   """ Returns the indecies of pids in pidVect """
   return np.where(np.isin(pidVect, pids))[0]
