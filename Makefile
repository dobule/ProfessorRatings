# Name: Ryan Gelston
# Description: Some basic scripts

clean: 
	rm *.pyc

createDirs:
	mkdir figures
	mkdir figures/correlations
	mkdir figures/knnAccuracy
	mkdir figures/tokenCount
	mkdir data
	mkdir data/correlations
	mkdir data/knnPredictions
	mkdir data/pids
	mkdir data/smilarityMatricies
	mkdir data/stats
	mkdir data/tokenCounts
	mkdir data/vectors

zipProject:
	zip TermProject-Gelston.zip \
		Report/Gelston-TermProject.pdf \
		Makefile \
		README.md \
		Counting.py \
		countTokens.py \
		createProfDict.py \
		createProfVectors.py \
		FfnnAlgorithm.py \
		FileNames.py \
		findCorrelations.py \
		findPids.py \
		getStatistics.py \
		KnnAlgorithm.py \
		PlotData.py \
		ReadFromFile.py \
		runFFNN.py \
		runKNN.py \
		Stats.py \
		VectorProcessing.py \
		WriteToFile.py \
		data/polyratings.sql \
		data/profDicts.pkl \
		data/professors.pkl \
		data/reviews.pkl \
		data/stopwords-mysql.txt \
		data/correlations/ \
		data/knnPredictions/ \
		data/pids/* \
		data/similarityMatricies/* \
		data/tokenCounts/* \
		data/vectors/* \
		figures/correlations/ \
		figures/knnAccuracy/ \
		figures/tokenCount/ 



countTokens: 
	python3 countTokens.py -ss -save
	python3 countTokens.py -ss -tup -save
	python3 countTokens.py -ss -stup -save

profVects:
	python3 createProfVectors.py -ss -minCount 40
	python3 createProfVectors.py -ss -tup -minCount 30
	python3 createProfVectors.py -ss -stup -minCount 40
	python3 createProfVectors.py -ss -minCount 40 -corr 45 0.45
	python3 createProfVectors.py -ss -tup -minCount 30 -corr 20 0.45
	python3 createProfVectors.py -ss -stup -minCount 40 -corr 50 0.46


correlations:
	python3 findCorrelations.py -ss -minCount 40 -save
	python3 findCorrelations.py -ss -tup -minCount 30 -save
	python3 findCorrelations.py -ss -stup -minCount 40 -save


ffnn-overall:
	python3 runFFNN.py -ss -minCount 40 -corr 45 0.45 -save
	python3 runFFNN.py -ss -tup -minCount 30 -corr 20 0.45 -save
	python3 runFFNN.py -ss -stup -minCount 40 -corr 50 0.46 -save
	python3 runFFNN.py -deep -ss -minCount 40 -corr 45 0.45 -save
	python3 runFFNN.py -deep -ss -tup -minCount 30 -corr 20 0.45 -save
	python3 runFFNN.py -deep -ss -stup -minCount 40 -corr 50 0.46 -save


knn-all:
	make knn-overall
	make knn-difficulty

knn-overall:
	make knn-overall-sin
	make knn-overall-tup
	make knn-overall-stup

knn-overall-corr:
	make knn-overall-sin-corr
	make knn-overall-tup-corr
	make knn-overall-stup-corr

knn-difficulty:
	make knn-difficulty-sin
	make knn-difficulty-tup
	make knn-difficulty-stup

knn-overall-sin-corr:
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -cos -save
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -tf -cos -save
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -tfidf -cos -save
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -pear -save
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -tf -pear -save
	python3 runKNN.py -ss -minCount 40 -corr 45 0.45 -tfidf -pear -save

knn-overall-tup-corr:
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -tf -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -tfidf -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -pear -save
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -tf -pear -save
	python3 runKNN.py -ss -tup -minCount 30 -corr 20 0.45 -tfidf -pear -save

knn-overall-stup-corr:
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -tf -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -tfidf -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -pear -save
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -tf -pear -save
	python3 runKNN.py -ss -stup -minCount 40 -corr 50 0.46 -tfidf -pear -save

knn-overall-sin:
	python3 runKNN.py -ss -minCount 40 -cos -save
	python3 runKNN.py -ss -minCount 40 -tf -cos -save
	python3 runKNN.py -ss -minCount 40 -tfidf -cos -save
	python3 runKNN.py -ss -minCount 40 -pear -save
	python3 runKNN.py -ss -minCount 40 -tf -pear -save
	python3 runKNN.py -ss -minCount 40 -tfidf -pear -save

knn-difficulty-sin:
	python3 runKNN.py	-d -ss -minCount 40 -cos -save
	python3 runKNN.py	-d -ss -minCount 40 -tf -cos -save
	python3 runKNN.py	-d -ss -minCount 40 -tfidf -cos -save
	python3 runKNN.py	-d -ss -minCount 40 -pear -save
	python3 runKNN.py	-d -ss -minCount 40 -tf -pear -save
	python3 runKNN.py	-d -ss -minCount 40 -tfidf -pear -save

knn-overall-tup:
	python3 runKNN.py -ss -tup -minCount 30 -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -tf -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -tfidf -cos -save
	python3 runKNN.py -ss -tup -minCount 30 -pear -save
	python3 runKNN.py -ss -tup -minCount 30 -tf -pear -save
	python3 runKNN.py -ss -tup -minCount 30 -tfidf -pear -save


knn-difficulty-tup:
	python3 runKNN.py	-d -ss -tup -minCount 30 -cos -save
	python3 runKNN.py	-d -ss -tup -minCount 30 -tf -cos -save
	python3 runKNN.py	-d -ss -tup -minCount 30 -tfidf -cos -save
	python3 runKNN.py	-d -ss -tup -minCount 30 -pear -save
	python3 runKNN.py	-d -ss -tup -minCount 30 -tf -pear -save
	python3 runKNN.py	-d -ss -tup -minCount 30 -tfidf -pear -save


knn-overall-stup:
	python3 runKNN.py -ss -stup -minCount 40 -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -tf -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -tfidf -cos -save
	python3 runKNN.py -ss -stup -minCount 40 -pear -save
	python3 runKNN.py -ss -stup -minCount 40 -tf -pear -save
	python3 runKNN.py -ss -stup -minCount 40 -tfidf -pear -save


knn-difficulty-stup:
	python3 runKNN.py	-d -ss -stup -minCount 40 -cos -save
	python3 runKNN.py	-d -ss -stup -minCount 40 -tf -cos -save
	python3 runKNN.py	-d -ss -stup -minCount 40 -tfidf -cos -save
	python3 runKNN.py	-d -ss -stup -minCount 40 -pear -save
	python3 runKNN.py	-d -ss -stup -minCount 40 -tf -pear -save
	python3 runKNN.py	-d -ss -stup -minCount 40 -tfidf -pear -save

