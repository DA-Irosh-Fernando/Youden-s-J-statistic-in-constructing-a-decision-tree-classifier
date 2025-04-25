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
=========================================================================================================
For each feature in the dataset, first it removes duplicates values and sort the data. Next, using each data point of each feature as a threshold, we calculate the J score. The data point with the highest J score is chosen as the cutoff point. The results are visualised using plots.



**3) Class: ProbabilityTree3Classifier.py**
======================================
This is the other core class which implement a symmetric probability decision tree with tree features and having the following methods according to the convention of the classifiers in machine learning. 
Note: this is a naive but a quick implementation of the concept for a set number of features. A better implementation for building a tree dynamically for any given number of features, has later been completed and will be made available later.

3.1) Method: fit(self,X,y)
--------------------------
Fit a training data set with X and y as described previously.

3.2) Method: predict(X_test)
--------------------
Predict the outcome using the test data set X-test and returns a binary vector corresponding to the outcome.

 3.2) method: predict_proba(X_test)
 ----------------------------------
Same as the above except it returns a vector corresponding to the outcome which are the actual probabilities.

**3) Class: ProbabilityTree2Classifier.py**
=====================================
Same as the above class except it build a simpler tree using only two features. 


**4) DecisionTreeClassifer_top_3_features.py**
=============================================
Uses sklearn’s DecisionTreeClassifier with the 3 features from the dataset having the highest J score to construct a Decision Tree Classifier. Outcome scores are generated using Gini, Entropy and Log-loss functions (lines 55-61: needs commenting and uncommenting the relevant instruction accordingly). Note that the outcome scores are written to a csv file to be used in ROC curve analysis using Stata statistical software later. However, there are phyton libraries including MLstatkit and roc_comparison which can be used to do a similar analysis.

**5) JscoreTreeClassifier_top_3_features.py**
==============================================
Finally, this is the main file that implements the proposed classification tree using the previously stated two core classes: 1) JscoreBinariser.py;  2) ProbabilityTree3Classifier.py. 
