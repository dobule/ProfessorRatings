# Name: Ryan Gelston (rgelston)
# Filename: ReadFromFile.py
# Assignment: Term Project
# Description: Contains functions that read data from various data files

import pickle
import os
import numpy as np
import FileNames as fn


PID_RAT_IDX = 0
DIFFICU_RAT_IDX = 1
OVERALL_RAT_IDX = 2


def prof_dicts():
   if not os.path.exists(fn.ProfDictsFile):
      print(fn.ProfDictsFile, "not found")
      print("Run 'python3 createProfDict.py'")
      exit()

   with open(fn.ProfDictsFile, 'rb') as f:
      profs = pickle.load(f)
   return profs


def stopwords(stmr=None):
   if not os.path.exists(fn.StopwordsFile):
      print(fn.StopwordsFile, "not found")
      print("Cannot continue without stopwords file")
      exit()

   with open(fn.StopwordsFile, 'r') as f:
      words = f.read()
   words = words.split()

   if type(stmr) != type(None):
      stemmedWords = []
      for word in words:
         stemmedWords.append(stmr.stem(word))
      words = stemmedWords
   
   return set(words)


def token_count(fileName, asTups=False):

   if not os.path.exists(fileName):
      return None

   with open(fileName, 'r') as f:
      rawText = f.read()

   lines = rawText.split('\n')
   lines = lines[1:-1]
   countTups = [line.split(',') for line in lines]
   countTups = [(t[0], int(t[1])) for t in countTups]

   if not asTups:
      countDict = {}
      for token, count in countTups:
         countDict[token] = count
      return countDict

   return countTups


def prof_token_dicts(filename):
   if not os.path.exists(filename):
      return None
   with open(filename, 'rb') as f:
      return pickle.load(f)


def pids_file(filename):
   if not os.path.exists(filename):
      return None
   return np.genfromtxt(filename,
                        dtype=int,
                        delimiter=',')


def vocab_from_word_count(countFile):
   if not os.path.exists(countFile):
      return None

   with open(countFile, 'w') as f:
      vocab = f.read()

   vocab = vocab.split('\n,')
   vocab = vocab[0::2]

   return vocab


def vocab_from_vect_file(vectorFile):
   if not os.path.exists(vectorFile):
      return None

   with open(vectorFile) as f:
      f.readline()
      f.readline()
      f.readline()
      f.readline()
      vocab = f.readline()

   vocab = vocab[:-1]
   vocab = vocab.split(',')

   return vocab[3:]


def pid_vect(vectorFile):
   if not os.path.exists(vectorFile):
      return None
   return np.genfromtxt(vectorFile, 
                        dtype=int,
                        delimiter=',',
                        skip_header=5, 
                        usecols=(PID_RAT_IDX))


def difficulty_rating_vect(vectorFile):
   if not os.path.exists(vectorFile):
      return None
   return np.genfromtxt(vectorFile,
                        dtype=float,
                        delimiter=',',
                        skip_header=5,
                        usecols=(DIFFICU_RAT_IDX))


def overall_rating_vect(vectorFile):
   if not os.path.exists(vectorFile):
      return None
   return np.genfromtxt(vectorFile,
                        dtype=float,
                        delimiter=',',
                        skip_header=5, 
                        usecols=(OVERALL_RAT_IDX))


def word_vects(vectorFile):
   if not os.path.exists(vectorFile):
      return None
   rawVects = np.genfromtxt(vectorFile, 
                            dtype=float,
                            delimiter=',',
                            skip_header=5)
   return rawVects[:,3:]


def knn_predictions(predFile):
   if not os.path.exists(predFile):
      return None
   return np.genfromtxt(predFile,
                        dtype=float,
                        delimiter=',')


def similarity_matrix(simMatFile):
   if not os.path.exists(simMatFile):
      return None
   return np.genfromtxt(simMatFile,
                        dtype=float,
                        delimiter=',')


def token_correlations(corrFile):
   if not os.path.exists(corrFile):
      return None
   with open(corrFile, 'r') as f:
      corrs = f.read()
   corrs = corrs.split('\n')
   corrs = corrs[1:-1] # Get rid of key on first line and last newline
   corrs = [cor.split(',') for  cor in corrs]

   result = []
   for cor in corrs:
      if cor[3] == 'nan':
         continue
      result.append((cor[0], cor[1], int(cor[2]), float(cor[3])))

   return result
