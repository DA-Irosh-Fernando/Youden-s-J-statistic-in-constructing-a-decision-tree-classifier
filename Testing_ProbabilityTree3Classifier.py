# -*- coding: utf-8 -*-
"""
Let us, your servants, see you work again; let our children see your glory. 
And may the Lord our God show us his approval and make our efforts successful. 
Yes, make our efforts successful!

Psalms 90:16-17 NLT.

Created on Mon Feb 12, 2025

@author: Irosh Fernando
"""


import numpy as np


#Create a training data set
#Testing a different dataset
X_train=np.array([[1,0,0,1,1,1,0,0],[0,1,0,1,0,1,0,0],[1,1,1,1,0,1,0,0]])
y_train=[1,0,1,1,0,1,0,0]

X_train=np.transpose(X_train)

#Create and train the tree
import ProbabilityTree3Classifier as pbt3
myTree=pbt3.ProbabilityTree3Classifier()
myTree.fit(X_train,y_train)

#Create a test data set for predicting 
X_test=[[0,1,1],[1,1,1],[0,0,1],[1,0,1]]


y_predict=myTree.predict(X_test)
y_predict_proba=myTree.predict_proba(X_test)
print(y_predict)
print(y_predict_proba)





