This repository contains the materials that supplement the paper titled “Youden's J statistic in constructing a decision tree classifier for medical diagnostic reasoning” published in the proceedings of the eHealth-2025 conference held in Lisbon Portugal. 
As you may go through the content of the paper, here is the order of the content in this repository that you may follow.

**1) Class: JscoreBinariser.py**
============================
This is the core class. It’s goal is to binarise a continuous dataset including the corresponding target values using the J score, using the following methods.

1.1) Method: derive_j(tn, fp, fn, tp): 
------------------------------
Derives the J score.
tn: true negatives.
fp: false positives.
fn: false negatives.
tp: true positives.
Returns: Youden's J statistic ( J score).

1.2) Method: derive_cutoffs(X,y)
--------------------------------
Generates the cutoff points for each feature using J score.  
X is 2-dimentinal array with n rows and m columns of feature set.
Y is a one-dimensional  array with n rows of target ( or outcome) values.
Returns: a row vector of size m consisting of the cut-off points for each column in X. 

1.3) Method:binarise(X,y)
-------------------------
binarise a continuous dataset including the target values.
parameters: as above and X consisting of continuous data.
Returns: a binarised version of X using the cutoff derived from the above-method.

1.3) Method:reduce_features(X,y,k)
----------------------------------
Reduces the number of columns m in X to k by eliminating columns of the features having low J scores. 
Returns: the reduced 2-dimentinal array with n rows and k columns.


**Dataset: Pima Indians Diabetes Database**
==========================================
Available via https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database. The data file is ‘diabetes.csv’ which needs to be saved and available for the following programs. 

**2)Plot_J_scores_and_cutoffs_using_the_max_J_score_for_each_featur_in_Pimpa_Indians_daibetes_data_se.py**
--------------------------------------------------------------------------------------------------------
For each feature in the dataset, first it removes duplicates values and sort the data. Next, using each data point of each feature as a threshold, we calculate the J score. The data point with the highest J score is chosen as the cutoff point. The results are visualised using plots.
