# Name: Ryan Gelston (rgelston)
# Filename: findPids.py
# Assignment: Term Project
# Description: Finds the pids of professors with only one review and with 
#  reviews with a total word count one standard deviation away from the 
#  mean review length.

import sys
import ReadFromFile as read
import WriteToFile as write
import FileNames as fn
import Stats as stat

MaxWordCount = stat.meanRevLen + stat.stdDevRevLen

def print_usage_message():
   print("python3 findPids.py [-h]")
   print("\t-h -- Print usage message")


def main():

   if '-h' in sys.argv:
      print_usage_message()
      exit()

   # Open profTokenDicts with raw word count
   profDicts = read.prof_dicts()

   singlePids = []
   smallLenPids = []

   # Iterate through prof token dicts
   for prof in profDicts:
      if len(prof['reviews']) == 1:
         singlePids.append(prof['pid'])
      total = 0
      for rev in prof['reviews']:
         total += len(rev['text'])
      if total <= MaxWordCount:
         smallLenPids.append(prof['pid'])
      

   print("Num singlePids:", len(singlePids))
   print("Num small pids:", len(smallLenPids))

   singlePids.sort()
   smallLenPids.sort()
      
   write.pids_file(singlePids, fn.PidsSingleRevFile)
   write.pids_file(smallLenPids, fn.PidsSmallRevLenFile)


if __name__=="__main__":
   main()
