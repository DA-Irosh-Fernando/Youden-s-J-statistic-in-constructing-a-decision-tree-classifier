# -*- coding: utf-8 -*-
"""
Blessed be your name Lord my God;
Please help me, guide me, and lead me.
With your hand, I will do mighty deeds to give glory to you and serve the world.
Amen.
Created on Mon Nov 25 19:15:55 2024
Edited on Sat 08 Feb 2025.


#---------
  
Let your servants see the wonderful things you can do for them.
    And let their children see your glory.
 Lord, our God, be kind to us.
    Make everything we do successful.
    Yes, make it all successful.
Psalm 90:16-17

08 Feb 2025

@author: Irosh Fernando
"""


import numpy as np
from sklearn import metrics

class JscoreBinariser:

    def __init__(self):
        pass

#Define a function to calculate J score and F score
    def derive_j(self,tn, fp, fn, tp):
        tpr=tp/(tp+fn)
        fpr=fp/(tn+fp)
        
        #ppv=tp/(tp+fp)
        j=tpr-fpr
        #f=2*(ppv*tpr)/(ppv+tpr)
        return j #, f  
        #uncomment the above lines to retrun f score as well

    def derive_cutoffs(self,X,y):
        self.J_scores=[] # 2D array to store J scores for each value in the data range of each feature
        self.J_max=[] # MaximumJ score for each feature
        self.Cutoffs=[] #cutoff points according to the best threshold for each clinical fesature
        for index in range(X.shape[1]):
        
            y_pred=X[:,index]
            
            #remove any duplicates
            cutoffs=np.sort(list(set(y_pred)))
            
            J=[]
            for c in cutoffs:
                 y_pred_binary = [1 if x >= c else 0 for x in y_pred]
                 tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred_binary).ravel()
                 j=self.derive_j(tn, fp, fn, tp) 
                 J.append(j)
            self.J_scores.append(J)    
            j_max=np.max(J)
            self.J_max.append(j_max)    
            best_threshold_J=cutoffs[np.argmax(J)]
            self.Cutoffs.append(best_threshold_J)     
        return self.Cutoffs, self.J_max


# Define a function to birarise using the cutoffs-
    def binarise(self,X,y):
        Cutoffs, Jmax =self.derive_cutoffs(X,y)
        for index in range(X.shape[1]):
            X[:,index]= [1 if x >= Cutoffs[index] else 0 for x in X[:,index]]
        return X

#Define a function to construct a new data set with n features having highest J score
# It seems dataframes has more advantages when building trees compared to 2D arrays
    def reduce_features(self,X,y,n):
        self.derive_cutoffs(X, y)
        indices = np.argpartition(self.J_max, -n)[-n:]

        #J_max=np.array(self.J_max)
        #top_n=J_max[indices]
        #Sort the array in descending order
        indices=indices[::-1]
        
        #top_n[::-1]
        
        X_new=[]
        
        for i in indices:
            X_new.append(X[:,i])
        
        #transform it to the standard form
        X_new=np.array(X_new).astype(int)
        X_new=np.transpose(X_new)
        return indices,X_new

