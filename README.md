# Predicting Professor Ratings From Reviews

This was a project of a data mining class I completed in college. 

## License

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished 
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.


## Miscellaneous Files

README.md -- You're reading it now.
Makefile -- Contains various recipies for creating directories, zipping 
   the project together, and running large batches of the KNN algorithm.

## Command Line Programs

The usage message for any of the command line programs can be found by using
the '-h' flag. For example: 'python3 runKNN.py -h'.

createProfDict.py -- Creates the profDicts.pkl data structure.
createProfVectors.py -- Creates professor token count vectors.
findCorrelations.py -- Finds pearson correlation between token pair count 
   ratios and rating value. Outputs a plot.
findPids.py -- Creates pids files, which hold the pids of professors with
   single reviews, and small reviews. 
getStatistics.py -- Finds some basic information about the distrobution of
   reviews and their sizes among professors. This also calculates the mean 
   absolute error of the naive prediction approach, which is always
   predicting the mean rating.
runFFNN.py -- Trains a feedforward neural network on professor vectors.
   Excludes single and small vectors to use as validation set.
runKNN.py -- Runs the KNN algorithm to predict rating for all professor
   vectors.

## Libraries

Counting.py -- Contains functions used in countTokens.py
FfnnAlgorithm.py -- Contains two FFNN topologies.
FileNames.py -- Contains constants and functions to hold and create filenames.
KnnAlgorithm.py -- Contains functions to run KNN and create the similarity 
   matrix.
PlotData.py -- Contains functions to create plots to visualize data.
ReadFromFile.py -- Reads various data structures from file.
Stats.py -- Contains functions used in getStatistics.py
VectorProcessing.py -- Contains functions concerned with vector creation
   and modification.
WriteToFile.py -- Writes various data structures to file.

## Data Directory

data/polyratings.sql -- SQL Dump of the datasets used.
data/profDicts.pkl -- Holds a list of profDicts, as described in the report.
data/professors.pkl -- Holds the professor table of the polyratings dataset
   as a list of dictionaries.
data/reviews.pkl -- Holds the review table of the polyratings dataset as a 
   list of dictionaries.
data/stopwords-mysql.txt -- Contains stopwords used during initial vector
   processing.

data/correlations/ -- Purposely left empty to reduce the size of the zip
   file. For a plot of these correlations refer to the report.
data/knnPredictions/ -- Directory included in zip, though contains no data.
   This directory holds predictions of professor vectors for various k values
   and will be populated with runs of runKNN.py. Predictions were excluded 
   from the zip file to reduce the files size.
data/pids/ -- Contans pids of professors with only one review and with small
   aggrigate reviews, as defined in the report.
data/similarityMatricies/ -- Contains similarity matricies for different
   vectors.
data/tokenCounts/ -- Contains token count files for various vector 
   configurations as well as profTokenDict pickled datastructures, which 
   are an intermediary datastructure for vector creation.
data/vectors/ -- Contians professor vector files.

## Figure Directory 

The directory structure for figures is included in the zip file, though
no figures are included. This decision was made in order to reduce the size of
the zip file. Instructions to create figures are below. 

figures/counts.png -- Plots the distrobution of review size, number of
   reviews per professor, and aggrigate review size for each professor.
   To create this file, run 'python3 getStatistics.py -save'.

figures/correlations/ -- Normally contains correlation plots. These are also
   found in the report. To create the run findCorrelations.py with the 
   '-save' flag. Note: If the correlations file hasn't been created
   beforehand, this will take a very long time to run, perhaps up to six
   hours.
figures/knnAccuracy/ -- Contains plots on the accuracy of a KNN run for all
   professor vectors, professor vectors with one review, and small professor
   vectors. To create these plots, run runKNN.py with the '-save' flag.
figures/tokenCount/ -- Contains plots showing the distrobution of tokens for 
   different token processing methods. To create this plot run 
   countTokens.py with the '-save' flag.
