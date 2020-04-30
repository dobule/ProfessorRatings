# Name: Ryan Gelston (rgelston)
# File: PlotData.py
# Assignment: Term Project
# Description: Contains functions to create plots.

import numpy as np
import matplotlib.pyplot as plt

import FileNames as fn

###########################################################
# Plotting Token Counts
###########################################################

def token_counts(rawTokens, revTokens, profTokens, 
      filename=None,
      title=None):

   rawCount = [c for t, c in rawTokens]
   revCount = [c for t, c in revTokens]
   profCount = [c for t, c in profTokens]

   rawCount.sort(reverse=True)
   revCount.sort(reverse=True)
   profCount.sort(reverse=True)

   plt.plot(rawCount)
   plt.plot(revCount)
   plt.plot(profCount)

   plt.yscale('log')

   if title is None:
      plt.title('Token Occurances')
   else:
      plt.title(title)

   plt.ylabel('# of occurances')
   plt.xlabel('Words in decending order')

   plt.legend(['# of times word appears in reviews',
               '# of reviews word appears in',
               '# of professors word appears in'],
               loc='upper right')

   if filename is None:
      plt.show()
   else:
      plt.savefig(filename)


###########################################################
# Plotting KNN error
###########################################################

def knn_error(preds, scores,
              title=None,
              subTitles=None, 
              idxToPlot=None,
              saveFile=None):
   """ Plots the error of various k values of a KNN run """

   # Plot multiple errors
   if idxToPlot is not None:
      fig, axs = plt.subplots(len(idxToPlot)+1, 1, constrained_layout=True)
      create_knn_error_sub_plot(axs[0], preds, scores,
         title="Error with all prof vects")
      for idx in range(len(idxToPlot)):
         subTitle = ""
         if subTitles is not None:
            subTitle = subTitles[idx]

         create_knn_error_sub_plot(
            axs[idx+1],
            preds[idxToPlot[idx],:],
            scores[idxToPlot[idx]],
            title=subTitle)

      fig.set_figheight(9)
      fig.set_figwidth(7)
      fig.subplots_adjust(hspace=0.7)
   else:
      fig, axs = plt.subplots(1,1)
      create_knn_error_sub_plot(axs[0], preds, scores)

   if title is not None:
      fig.suptitle(title, wrap=True)

   # Save or show plot
   if saveFile is None:
      plt.show()
   else:
      plt.savefig(saveFile)


def create_knn_error_sub_plot(ax, preds, scores, title=None):

   errors = (preds.T - scores).T
   absErr = np.absolute(errors)
   means = np.nanmean(absErr, axis=0)
   stdDivs = np.nanstd(absErr, axis=0)
   kVals = np.arange(means.shape[0], dtype=int) + 1

   minMeanIdx = np.where(means == np.amin(means))
   if type(minMeanIdx) is np.ndarray:
       minMeanIdx.sort()
       minMeanIdx = minMeanIdx[0]

   minMean = means[minMeanIdx]
   minStd = stdDivs[minMeanIdx]
   minKVal = kVals[minMeanIdx]

   if title is not None:
      ax.set_title(title, wrap=True)

   ax.set_ylabel('Abs Error')
   ax.set_xlabel('K Value')

   # TODO: Change text box position to upper right
   textbox = dict(boxstyle='round', 
                  facecolor='white', 
                  alpha=0.5)
   minStr = '\n'.join((
               r'$\mathrm{k}=%d$' % minKVal,
               r'$\mu=%.2f$' % minMean,
               r'$\sigma=%.2f$' % minStd))
   ax.text(50, 1, s=minStr, 
           fontsize=12,
           verticalalignment='top',
           bbox=textbox)

   ax.errorbar(kVals,
               means,
               stdDivs,
               marker='',
               linewidth=1,
               alpha=0.5,
               color='b')
   ax.plot(kVals, means,
           marker='',
           linewidth=1,
           color='b')

   ax.errorbar(minKVal,
               minMean,
               minStd,
               marker='',
               linewidth=1.5,
               alpha=1,
               color='r')
   ax.plot(minKVal, minMean,
           marker='o',
           color='r')


def create_knn_error_title(argv):
   options = []
   if '-ss' in argv:
      options.append("Stemmed & Stopped")
  
   if '-tup' in argv:
      options.append("Tups")
   elif '-stup' in argv:
      options.append("Sin & Tups")
   else:
      options.append("Singles")

   if '-tf' in argv:
      options.append("Tf Vect")
   elif '-tfidf' in argv:
      options.append("Tf-Idf Vect")
   else:
      options.append("Count Vect")

   if '-cos' in argv:
      options.append("Cos Sim")
   elif '-pear' in argv:
      options.append("Pear Corr")
   else:
      options.append("Inv Euc")

   if '-corr' in argv:
      cntStr = str(argv[argv.index('-corr') + 1])
      corStr = str(argv[argv.index('-corr') + 2])
      options.append("Min Pair Count " + cntStr)
      options.append("Min Pear Corr " + corStr)
   elif '-minCount' in argv:
      cntStr = str(argv[argv.index('-minCount') + 1])
      options.append("Min Count " + cntStr)

   return "KNN Error: " + ', '.join(options)
 
###########################################################
# Plotting Review Length Distrobution
###########################################################

def plot_word_review_count(revLen, wordsPerProf, revPerProf, save=False):
   """ Plots information about word and review distrobutions """

   revLen.sort()
   # revLen = revLen[::-1]
   wordsPerProf.sort()
   # wordsPerProf = wordsPerProf[::-1]
   revPerProf.sort()
   # revPerProf = revPerProf[::-1]

   fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)

   fig.suptitle("Count Plots")

   textbox = dict(boxstyle='round', facecolor='white', alpha=0.5)

   ax0.plot(revLen)
   ax0.set_title("# of words in each review")
   ax0.set_xlabel("Reviews")
   ax0.set_ylabel("# of Words")
   ax0.set_yscale('log')

   revLenStatStr = stats_text_str(revLen)
   ax0.text(0.05, 0.95, revLenStatStr, fontsize=12, transform=ax0.transAxes,
      verticalalignment='top', bbox=textbox)

   ax1.plot(wordsPerProf)
   ax1.set_title("# of words in each professor's reviews")
   ax1.set_xlabel("Professors")
   ax1.set_ylabel("# of Words")
   ax1.set_yscale('log')

   WPPStatStr = stats_text_str(wordsPerProf)
   ax1.text(0.05, 0.95, WPPStatStr, fontsize=12, transform=ax1.transAxes,
      verticalalignment='top', bbox=textbox)

   ax2.plot(revPerProf)
   ax2.set_title("# of reviews for each professor")
   ax2.set_xlabel("Professors")
   ax2.set_ylabel("# of Reviews")
   ax2.set_yscale('log')

   RPPStatStr = stats_text_str(revPerProf)
   ax2.text(0.05, 0.95, RPPStatStr, fontsize=12, transform=ax2.transAxes,
      verticalalignment='top', bbox=textbox)

   fig.set_figwidth(6.5)
   fig.set_figheight(8)
   fig.subplots_adjust(left=0.1, hspace=0.5)

   if save:
      plt.savefig(fn.CountFigureFile)
   else:
      plt.show()


def stats_text_str(data):
   return '\n'.join((
      r'$\mathrm{n}=%d$' % data.shape[0],
      r'$\mu=%.2f$' % data.mean(),
      r'$\sigma=%.2f$' % data.std()))


###########################################################
# Plotting FFNN Error
###########################################################

def ffnn_error(hist, title=None, filename=None):
   plt.plot(hist.history['loss'])
   plt.plot(hist.history['val_loss'])
   plt.ylabel('Mean Absolute Error')
   plt.xlabel('Epoch')
   plt.legend(['Train', 'Validation'], loc='upper right')

   if title is not None:
      plt.title(title, wrap=True)

   if filename is not None:
      plt.savefig(filename)
   else:
      plt.show()


def ffnn_error_title(argv):
   options = []

   if '-deep' in argv:
      options.append("Deep Net")
   else:
      options.append("Single Layer Net")

   if '-ss' in argv:
      options.append("Stemmed & Stopped")
  
   if '-tup' in argv:
      options.append("Tups")
   elif '-stup' in argv:
      options.append("Sin & Tups")
   else:
      options.append("Singles")

   if '-corr' in argv:
      cntStr = str(argv[argv.index('-corr') + 1])
      corStr = str(argv[argv.index('-corr') + 2])
      options.append("Min Pair Count " + cntStr)
      options.append("Min Pear Corr " + corStr)
   elif '-minCount' in argv:
      cntStr = str(argv[argv.index('-minCount') + 1])
      options.append("Min Count " + cntStr)

   return "FFNN Error: " + ', '.join(options)


###########################################################
# Plotting Token/score correlation
###########################################################

def tuple_pair_score_correlation(corrTups, 
                                 title=None,
                                 saveFile=None):

   numOccurances = np.array([cor[2] for cor in corrTups],
                            dtype=int)
   pearCorrScore = np.abs(np.array([cor[3] for cor in corrTups],
                                   dtype=float))

   fig, ax = plt.subplots(1,1, constrained_layout=True)
   ax.scatter(numOccurances, pearCorrScore,
           marker='.',
           alpha=0.05)
   ax.set_xlabel("Occurances")
   ax.set_ylabel("Absolute Pearson Correlation")
   ax.set_xscale('log')

   if title is not None:
      fig.suptitle(title)
   fig.set_figheight(6)
   fig.set_figwidth(8)

   if saveFile is None:
      plt.show()
   else:
      plt.savefig(saveFile)

def create_token_pair_score_correlation_name(argv):
   options = []

   if '-ss' in argv:
      options.append("Stemmed & Stopped")

   if '-tup' in argv:
      options.append("Tuples")
   elif '-stup' in argv:
      options.append("Sin & Tups")
   else:
      options.append("Singletons")

   return "Token pair correlation: " + ', '.join(options)

def plot_word_correlation(corrScore):

   corrScore = np.abs(corrScore)
   corrScore.sort()
   corrScore = corrScore[::-1]

   plt.plot(corrScore)
   plt.xlabel("# words with decreasing absolute correlation")
   plt.ylabel("Absolute Pearson Correlation Score")
   plt.show()


def plot_word_count_correlation(count, corr):

   corr = np.abs(corr)
   plt.scatter(count, corr,
               marker=".",
               alpha=0.3)
   plt.xlabel("Word count")
   plt.ylabel("Absolute Pearson Correlation")
   plt.xscale('log')
   plt.show()


   
