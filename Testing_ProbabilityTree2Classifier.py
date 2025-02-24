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
X_train=np.array([[1,0,0,1,1,1,0,0],[0,1,0,1,0,1,0,0]])
X_train=np.transpose(X_train)
y_train=[1,1,0,1,0,1,0,0]

#Create and train the tree
import ProbabilityTree2Classifier as pbt2
myTree2=pbt2.ProbabilityTree2Classifier()
myTree2.fit(X_train,y_train)
#print the probabilities y=1 in each branch 
print(" Probabilities y=1: ",myTree2.P1)

#print the probabilities y=01 in each branch 
print(" Probabilities y=0: ",myTree2.P0)

print ("Sum of the probabilities :", sum(myTree2.P0)+sum(myTree2.P1))




#Create a test data set for predicting 
X_test=[[0,1],[1,1],[0,0],[1,0]]
y_predict=myTree2.predict(X_test)
print(y_predict)

y_predict_proba=myTree2.predict_proba(X_test)
print(y_predict_proba)
