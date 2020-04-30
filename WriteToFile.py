# Name: Ryan Gelston (rgelston)
# Filename: WriteToFile.py
# Assignment: Term Project
# Description: Outputs various data structures to a file

import numpy as np

def token_count(tokenCounter, outFile):
   """ Writes the token counts to a csv file """

   if type(tokenCounter) != list:
      tokenCounter = [(k, v) for k, v in tokenCounter.items()]
   
   tokenCounter.sort(key=lambda t: (t[1], t[0]), reverse=True)

   with open(outFile, 'w') as f:
      f.write("%d # Number of token counts\n" % (len(tokenCounter)))
      for tup in tokenCounter:
         f.write("%s, %d\n" % tup)


def prof_vects(profVects, pidsNotIncluded, tokenSchema, outFileName):
   """ Writes profVects to outFileName """

   f = open(outFileName, 'w')
   f.write("%d # Number of vectors\n" % (len(profVects)))
   f.write("%d # Vector Length\n" % (len(tokenSchema) + 3))
   f.write("%d # Num pids not included\n" % len(pidsNotIncluded))
   f.write(','.join([str(pid) for pid in pidsNotIncluded]) 
            + " # Pids not included due to zero vectors\n")
   f.write("%s,%s,%s,%s\n" % 
      ("pid", "rating_difficulty", "rating_overall", ','.join(tokenSchema)))
      
   for vect in profVects:
      f.write("%d,%f,%f,%s\n" % 
                  (vect['pid'],
                   vect['rating_difficulty'],
                   vect['rating_overall'], 
                   ','.join([str(c) for c in vect['token_vect']])))
   f.close()


def similarity_matrix(simMat, outFile):
   np.savetxt(outFile,
              simMat,
              fmt='%f',
              delimiter=',',
              newline='\n')


def knn_predictions(preds, outFile):
   np.savetxt(outFile,
              preds,
              fmt='%f',
              delimiter=',',
              newline='\n')


def pids_file(pids, outFile):
   with open(outFile, 'w') as f:
      f.write(','.join([str(pid) for pid in pids]))


def token_correlations(corrTups, outFile):
   with open(outFile, 'w') as f:
      f.write("token1,token2,num_occurances,pearson_correlation\n")
      f.write('\n'.join(
                  [','.join([cor[0], cor[1], str(cor[2]), str(cor[3])])
                     for cor in corrTups]))
