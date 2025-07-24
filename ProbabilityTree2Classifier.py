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
#------------ Building the tree ---------------------------
#==========================================================
class ProbabilityTree2Classifier:
    def __init__(self):
        self.T=['11','10','01','00'] # Each branch
    
    def fit(self,X,y): 
        n=X.shape[0] #Number of data points in the training dataset (i.e., number of rows)
        index=list(range(0,n))
        df=pd.DataFrame(X,index=index, columns=['Feature1','Feature2'])
        df['Target']=y
        
        T11=df['Target'][df['Feature1']==1][df['Feature2']==1]
        T10=df['Target'][df['Feature1']==1][df['Feature2']==0]
        
        T01=df['Target'][df['Feature1']==0][df['Feature2']==1]
        T00=df['Target'][df['Feature1']==0][df['Feature2']==0]
        
        
        #Calculate probabilities for the outcome y=1
        #===========================================
        self.P1=[]
        self.P1.append(T11.sum()/n)
        self. P1.append(T10.sum()/n)
        self.P1.append(T01.sum()/n)
        self.P1.append(T00.sum()/n)
        
        
        #Calculate the probabilities for the outcome y=0
        #================================================
        
        self.P0=[]
        self.P0.append((len(T11)-T11.sum())/n)
        self.P0.append((len(T10)-T10.sum())/n)
        self.P0.append((len(T01)-T01.sum())/n)
        self.P0.append((len(T00)-T00.sum())/n)
        

        
    def predict(self,X_test):   
        T=np.transpose(self.T)
        T=list(T) #it has to be a list inorder to use key() function: index=T.index(key)
        y_predict=[]
        
        for i in range(len(X_test)):
            key=''
            for digit in X_test[i]:
                key+=str(digit)
                
            #print(key)
        
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
    
