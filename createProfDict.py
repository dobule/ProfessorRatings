# Name: Ryan Gelston
# Assignment: Final Project
# Filename: createProfDict.py
# Description: Creates a list of professor dictionaries that contain a 
#  list of reviews for that professor.
#
# Each prof dictionary has the following format:
#
#  {'pid': int,
#   'first_name': str,
#   'last_name': str,
#   'department': str,
#   'rating_difficulty': float,
#   'rating_overall': float,
#   'reviews': [{'class_name': str
#                'class_standing': str
#                'reason_taking': str
#                'grade_recieved': str
#                'date_posted': datetime
#                'review': [str, str, str,...]},
#                ...
#              ]
#   }
#
# The text of each review is tokenized and set to lowercase.


import pickle
import string
import sys
from nltk.tokenize import RegexpTokenizer

def print_help_message():
   print("python3 createProfDict.py [-h]")
   print("\t-h -- Print help message")


def main():

   if '-h' in sys.argv:
      print_help_message()

   with open('./data/reviews.pkl', 'rb') as f:
      reviews = pickle.load(f)

   with open('./data/professors.pkl', 'rb') as f:
      professors = pickle.load(f)

   profs = aggrigate_reviews(professors, reviews)
   profs = remove_bad_entries(profs)

   with open('./data/profDicts.pkl', 'wb') as f:
      pickle.dump(profs, f)


def aggrigate_reviews(profs, reviews):
   """ Tokenizes and aggrigates reviews by professor
         profs -- list of professors from professors.pkl
         reviews -- raw reviews from reviews.pkl

         Returns a list of dictionaries, with each dictionary
         representing the aggrigated reviews of a professor.
   """
  
   tknzr = RegexpTokenizer("[a-zA-Z][a-zA-Z']*[a-zA-Z]")
   professors = create_prof_dict(profs)

   for r in reviews:
      if r['pid'] not in professors.keys():
         continue
      if type(r['content']) is not str:
         continue

      prof = professors[r['pid']]

      # Assure that the prof's entry has these fields 
      if 'rating_difficulty' not in prof.keys():
         prof['rating_difficulty'] = r['rating_difficulty']
      if 'rating_overall' not in prof.keys():
         prof['rating_overall'] = r['rating_overall']

      # Create dictionary for review
      revDict = {}
      revDict['class_name'] = r['class_name']
      revDict['class_standing'] = r['class_standing']
      revDict['reason_taking'] = r['reason_taking']
      revDict['grade_received'] = r['grade_received']
      revDict['date_posted'] = r['date_posted']

      # Tokenize and make all tokens in the review lowercase
      rev = tknzr.tokenize(r['content'])
      rev = list(map(lambda s: s.lower(), rev))
      revDict['text'] = rev

      # Append the review to the professors review list
      prof['reviews'].append(revDict)

   return list(professors.values())
      

def create_prof_dict(profs):
   """ Creates a dictionary of professors with key being their pid """

   professors = {}

   for p in profs:
      if (p['pid'] not in professors.keys()
          and p['last_name'] != None):
         professors[p['pid']] = p
         professors[p['pid']]['reviews'] = []
   
   return professors


def remove_bad_entries(profs):

   newProfs = []

   for p in profs:
      if not is_prof_valid(p):
         continue
      newProfs.append(p)

   return newProfs

def is_prof_valid(prof):

   if (prof['last_name'] == None
       or prof['department'] == None 
       or type(prof['rating_difficulty']) != float
       or type(prof['rating_overall']) != float):
      return False
   
   hasText = False

   for rev in prof['reviews']:
      if rev['text'] != []:
         hasText = True
         break

   return hasText

   







if __name__=="__main__":
   main()
