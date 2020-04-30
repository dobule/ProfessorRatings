# Name: Ryan Gelston (rgelston)
# Filename: FileNames.py
# Assignment: Term Project
# Description: Contains constant filenames and directory names

# File name creation functions rely on command line arguments, which are as
# follows:
#
#  DEFAULT: Do not stem or remove stopwords
#  -ss -- Stem and remove stopwords
#
#  DEFAULT: Only use singletons as tokens
#  -tup -- Use tuples from reviews as tokens
#  -stup -- Use singletons and tuples from reviews as tokens
#
#  DEFUALT: Use raw token count vectors
#  -tf -- Use term frequency vector 
#  -tfidf -- Use tf-idf vector
#
#  DEFAULT: Use overall ratings 
#  -d -- Set the ratings vector to the difficulty ratings
#
#  DEFAULT: Use inverse euclidean distance as similarity functions
#  -cos -- Use cosine similarity as similarity function
# 
#  -minCount <int> -- Only include words that appear in at least <int> reviews


DataDir = "./data/"
PredictionsDir = DataDir + "knnPredictions/"
SimilarityMatDir = DataDir + "similarityMatricies/"
TokenCountDir = DataDir + "tokenCounts/"
VectorDir = DataDir + "vectors/"
PidsDir = DataDir + "pids/"
CorrelationsDir = DataDir + "correlations/"

PidsSingleRevFile = PidsDir + "pids_single_rev.csv"
PidsSmallRevLenFile = PidsDir + "pids_small_rev_len.csv"

ProfDictsFile = DataDir + "profDicts.pkl"
StopwordsFile = DataDir + "stopwords-mysql.txt"


FigureDir = "./figures/"
TokenCountFiguresDir = FigureDir + "tokenCount/"
KnnAccuracyFiguresDir = FigureDir + "knnAccuracy/"
CorrelationFiguresDir = FigureDir + "correlations/"
FfnnErrorFiguresDir = FigureDir + "ffnnError/"
CountFigureFile = FigureDir + "counts.png"


def create_token_count_names(argv):
   """ Returns a tuple of three file names, one for raw count, the next for
       review count and the third for prof count.
   """
   optionsPart = create_token_options_name_part(argv) + ".csv"
   rawName = TokenCountDir + "count_Raw" + optionsPart
   revName = TokenCountDir + "count_Rev" + optionsPart
   profName = TokenCountDir + "count_Prof" + optionsPart
   return (rawName, revName, profName)


def create_prof_token_dict_name(argv):
   """ Returns the name of the profTokenDict"""
   return (TokenCountDir 
            + "profTokenDict" 
            + create_token_options_name_part(argv)
            + ".pkl")


def create_prof_vect_name(argv, includeCorr=True):
   """ Returns the name of profVect """
   return (VectorDir 
            + "profVect" 
            + create_token_options_name_part(argv) 
            + create_vector_options_name_part(argv, includeCorr)
            + ".csv")


def create_count_plot_name(argv):
   return (TokenCountFiguresDir
            + "token_count"
            + create_token_options_name_part(argv)
            + ".png")


def create_knn_accuracy_plot_name(argv):
   return(KnnAccuracyFiguresDir
           + "knn_accuracy"
           + create_token_options_name_part(argv)
           + create_vector_options_name_part(argv, True)
           + create_knn_pred_name_part(argv)
           + ".png")


def create_sim_mat_name(argv):
   return (SimilarityMatDir
            + "simMat"
            + create_token_options_name_part(argv)
            + create_vector_options_name_part(argv, True)
            + create_knn_pred_name_part(argv)
            + ".csv")


def create_preds_name(argv):
   return (PredictionsDir
            + "knnPred"
            + create_token_options_name_part(argv)
            + create_vector_options_name_part(argv,True)
            + create_knn_pred_name_part(argv)
            + ".csv")


def create_correlations_name(argv):
   return (CorrelationsDir
            + "correl"
            + create_token_options_name_part(argv)
            + create_vector_options_name_part(argv)
            + ".csv")

def create_correlations_plot_name(argv):
   return (CorrelationFiguresDir
            + "correl"
            + create_token_options_name_part(argv)
            + create_vector_options_name_part(argv)
            + ".png")

def create_ffnn_plot_name(argv):

   if '-deep' in argv:
      ffnnPart = "_deep"
   else:
      ffnnPart = "_single"

   return (FfnnErrorFiguresDir
            + "ffnn"
            + ffnnPart
            + create_token_options_name_part(argv)
            + create_vector_options_name_part(argv, True)
            + ".png")

def create_token_options_name_part(argv):
   
   tknNamePart = "Sin"
   if '-tup' in argv:
      tknNamePart = "Tup"
   elif '-stup' in argv:
      tknNamePart = "SinTup"

   suffix = tknNamePart

   if '-ss' in argv:
      suffix += '_StmStp'

   return '_' + suffix


def create_vector_options_name_part(argv, includeCorr=False):

   namePart = ""
   if '-minCount' in argv:
      namePart += "_minCount_" + argv[argv.index('-minCount') + 1]

   if includeCorr and '-corr' in argv:
      corIdx = argv.index('-corr')
      minCnt = str(int(argv[corIdx + 1]))
      minScore = str(float(argv[corIdx + 2]))
      minScore = minScore.strip(' .0')
      namePart += "_corr_" + minCnt + "_" + minScore

   return namePart


def create_knn_pred_name_part(argv):

   ratNamePart = "Over"
   if '-d' in argv:
      ratNamePart = "Diff"

   vecNamePart = "Raw"
   if '-tf' in argv:
      vecNamePart = "Tf"
   elif '-tfidf' in argv:
      vecNamePart = "TfIdf"

   simNamePart = "InvEuc"
   if '-cos' in argv:
      simNamePart = "CosSim"
   elif '-pear' in argv:
      simNamePart = "PeaCor"

   wgtNamePart = "Wgted"
   if '-unweighted' in argv:
      wgtNamePart = "UnWgt"

   return '_' + '_'.join([ratNamePart,
                          vecNamePart,
                          simNamePart,
                          wgtNamePart])
