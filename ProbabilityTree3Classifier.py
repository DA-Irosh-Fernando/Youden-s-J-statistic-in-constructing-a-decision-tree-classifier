# -*- coding: utf-8 -*-
"""
Let us, your servants, see you work again; let our children see your glory. 
And may the Lord our God show us his approval and make our efforts successful. 
Yes, make our efforts successful!

Psalms 90:16-17 NLT.

Created on Mon Feb 12, 2025

@author: Irosh Fernando
"""

import pandas as pd
import numpy as np



#==========================================================
#------------ Probability tree class-----------------------
#==========================================================

class ProbabilityTree3Classifier:
    def __init__(self):
        #All possible permutations of the 3 features
        self.T=['111','110','101','100','011','010','001','000'] # Each branch


    def fit(self,X,y):
        n=X.shape[0] #Number of data points in the training dataset (i.e., number of rows)
        
        index=list(range(0,n))
        df=pd.DataFrame(X,index=index, columns=['Feature1','Feature2','Feature3'])
        df['Target']=y


        T111=df['Target'][df['Feature1']==1][df['Feature2']==1][df['Feature3']==1]
        T110=df['Target'][df['Feature1']==1][df['Feature2']==1][df['Feature3']==0]
        T101=df['Target'][df['Feature1']==1][df['Feature2']==0][df['Feature3']==1]
        T100=df['Target'][df['Feature1']==1][df['Feature2']==0][df['Feature3']==0]


        T011=df['Target'][df['Feature1']==0][df['Feature2']==1][df['Feature3']==1]
        T010=df['Target'][df['Feature1']==0][df['Feature2']==1][df['Feature3']==0]
        T001=df['Target'][df['Feature1']==0][df['Feature2']==0][df['Feature3']==1]
        T000=df['Target'][df['Feature1']==0][df['Feature2']==0][df['Feature3']==0]



        #Calculate probabilities for the outcome y=1
        #===========================================
        self.P1=[]
        self.P1.append(T111.sum()/n)
        self.P1.append(T110.sum()/n)
        self.P1.append(T101.sum()/n)
        self.P1.append(T100.sum()/n)
        
        self.P1.append(T011.sum()/n)
        self.P1.append(T010.sum()/n)
        self.P1.append(T001.sum()/n)
        self.P1.append(T000.sum()/n)
        
        #Calculate the probabilities for the outcome y=0
        #================================================
        
        self.P0=[]
        self.P0.append((len(T111)-T111.sum())/n)
        self.P0.append((len(T110)-T110.sum())/n)
        self.P0.append((len(T101)-T101.sum())/n)
        self.P0.append((len(T100)-T100.sum())/n)

        self.P0.append((len(T011)-T011.sum())/n)
        self.P0.append((len(T010)-T010.sum())/n)
        self.P0.append((len(T001)-T001.sum())/n)
        self.P0.append((len(T000)-T000.sum())/n)


    def predict(self,X_test):
        T=np.transpose(self.T)
        T=list(T) #it has to be a list inorder to use key() function: index=T.index(key)
       
        y_predict=[]
        
        for i in range(len(X_test)):
            key=''
            for digit in X_test[i]:
                key+=str(digit)
      
            index=T.index(key)
            y_predict.append(1 if self.P1[index] >self.P0[index] else 0)

        return y_predict
    
    
    def predict_proba(self,X_test):
        T=np.transpose(self.T)
        T=list(T) #it has to be a list inorder to use key() function: index=T.index(key)
       
        y_predict_proba=[]
        
        for i in range(len(X_test)):
            key=''
            for digit in X_test[i]:
                key+=str(digit)
      
            index=T.index(key)
            y_predict_proba.append(self.P1[index])
        return y_predict_proba
    
    





