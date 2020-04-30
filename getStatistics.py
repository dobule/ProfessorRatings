# Name: Ryan Gelston (rgelston)
# Filename: getStatistics.py
# Assignment: Term Project
# Description: Calculates basic statistics about the dataset

import sys
import numpy as np
import Stats as st
import ReadFromFile as read
import FileNames as fn
import PlotData as plot

def print_usage_message():
   print("python3 getStatistics.py [-save] [-h]")
   print("\t-save -- Saves the plot produced by this program to "
            + fn.CountFigureFile)
   print("\t-h -- Print usage message")

def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   profDicts = read.prof_dicts()

   revLens = st.rev_len_arr(profDicts)
   print("Number of Reviews:", revLens.shape[0])
   print("Mean review length:", revLens.mean())
   print("Std Dev review length:", revLens.std())
   print()

   numRevsProf = st.num_revs_profs(profDicts)
   print("Number of professors:", numRevsProf.shape[0])
   print("Mean num reviews per prof:", numRevsProf.mean())
   print("Std Dev num revews per prof:", numRevsProf.std())
   print()

   profRevLen = st.profs_revs_len(profDicts)
   print("Mean tokens per prof:", profRevLen.mean())
   print("Std Dev tokens per prof:", profRevLen.std())
   print()


   overRats = np.array([prof['rating_overall'] for prof in profDicts], 
                       dtype=float)
   diffRats = np.array([prof['rating_difficulty'] for prof in profDicts], 
                       dtype=float)

   overRatMean = overRats.mean()
   diffRatMean = diffRats.mean()

   print("Overall ratings mean:", overRatMean)
   print("Overall ratings std dev:", overRats.std())
   print("Difficulty ratings mean:", diffRatMean)
   print("Difficulty ratings std dev:", diffRats.std())
   print()

   overMeanDiff = overRats - overRatMean
   overMeanDiff = np.abs(overMeanDiff)
   diffMeanDiff = diffRats - diffRatMean
   diffMeanDiff = np.abs(diffMeanDiff)

   print("Nieve approach to prediction: Guessing the Mean")
   print("All profs")
   print("Overall absolute error mean:", overMeanDiff.mean())
   print("Overall absolute error std div:", overMeanDiff.std())
   print("Difficulty absolute error mean:", diffMeanDiff.mean())
   print("Difficulty absolute error std div:", diffMeanDiff.std())
   print()

   oneRevPids = set(read.pids_file(fn.PidsSingleRevFile))
   oneOverRats = np.array([prof['rating_overall'] for prof in profDicts
                            if prof['pid'] in oneRevPids],
                          dtype=float)
   oneOverDiff = np.abs(oneOverRats - oneOverRats.mean())

   print("Profs with one review")
   print("One review absolute error mean:", oneOverDiff.mean())
   print("One review absolute error std div:", oneOverDiff.std())
   print()


   smallRevPids = set(read.pids_file(fn.PidsSmallRevLenFile))
   smallOverRats = np.array([prof['rating_overall'] for prof in profDicts
                            if prof['pid'] in smallRevPids],
                          dtype=float)
   smallOverDiff = np.abs(smallOverRats - smallOverRats.mean())

   print("Profs with short reviews")
   print("Small review absolute error mean:", smallOverDiff.mean())
   print("small review absolute error std div:", smallOverDiff.std())
   print()
 
   save = False
   if '-save' in sys.argv:
      save = True

   plot.plot_word_review_count(
      revLens,  
      profRevLen,
      numRevsProf,
      save=save)

if __name__=="__main__":
   main()
