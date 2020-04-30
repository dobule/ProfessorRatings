# Name: Ryan Gelston (rgelston)
# Filename: Stats.py
# Assignment: Term Project
# Description: Calculates basic statistical information about the profDicts

import numpy as np

###########################################################
# Constants (Canned info about dataset)
# Run 'python3 getStatistics.py' for source 
###########################################################

numReviews = 65647
meanRevLen = 97.45619754139565
stdDevRevLen = 148.0694605270395

numProffesors = 2471
meanRevsPerProf = 26.566976932416026
stdDevRevsPerProf = 33.87400575681629


def num_profs(profDict):
   return len(profDict)


def num_revs_profs(profDict):
   numReviews = []
   for prof in profDict:
      numReviews.append(len(prof['reviews']))
   return np.array(numReviews, dtype=int)


def rev_len_arr(profDict):
   lens = []
   for prof in profDict:
      for rev in prof['reviews']:
         lens.append(len(rev['text']))
   return np.array(lens, dtype=int)


def profs_pid(profDict):
   pids = []
   for prof in profDict:
      pids.append(prof['pid'])
   return np.array(pids, dtype=int)


def profs_revs_len(profDict):
   profsLen = []
   for prof in profDict:
      revsLen = 0
      for rev in prof['reviews']:
         revsLen += len(rev['text'])
      profsLen.append(revsLen)
   return np.array(profsLen, dtype=int)


def profs_with_one_review(profDict):
   profPids = []
   for prof in profDict:
      if len(prof['reviews']) == 1:
         profPids.append(prof['pid'])
   return np.array(profPids)


def pids_value_dict(profDict, value):
   """ Creates a dictionary with the key being 'pid' and the values being the 
       value in a prof entry with key 'value'.
   """
   pidsDict = {}
   for prof in profDict:
      pidsDict[prof['pid']] = prof[value]
   return pidsDict


###########################################################
# Correlation Functions
###########################################################

def pearson_correlation(vect1, vect2):
   meanV1 = vect1.mean()
   meanV2 = vect2.mean()
   adjV1 = vect1 - meanV1
   adjV2 = vect2 - meanV2 
   return (np.sum(adjV1 * adjV2) 
            / ((np.sum(adjV1 ** 2) * np.sum(adjV2 ** 2)) ** 0.5))


def proportion_vector(vect1, vect2, ratVect):

   idx = 0
   idxList = []

   # Find indecies where token counts for both vects are non-zero
   for tc1, tc2 in zip(vect1, vect2):
      if tc1 != 0 and tc2 != 0:
         idxList.append(idx)
      idx += 1

   if idxList == []:
      return (None, None)

   return ((vect1[idxList] / vect2[idxList]), ratVect[idxList])


def find_correlations(tokenVects, ratingVect, vocabVect):
   """ Returns list of tuples (tok1, tok2, numOccurances, corrScore) """
   corrTups = []
   for tokIdx1 in range(len(vocabVect)):
      for tokIdx2 in range(tokIdx1+1, len(vocabVect)):
         propVect, ratVect = proportion_vector(
                                 tokenVects[:,tokIdx1],
                                 tokenVects[:,tokIdx2],
                                 ratingVect)
         if propVect is not None:
            corrTups.append((vocabVect[tokIdx1],
                             vocabVect[tokIdx2],
                             propVect.shape[0],
                             pearson_correlation(propVect, ratVect)))
   corrTups.sort(
      key=(lambda t: (np.absolute(t[3]), t[2], t[0], t[1])),
      reverse=True)
   return corrTups

