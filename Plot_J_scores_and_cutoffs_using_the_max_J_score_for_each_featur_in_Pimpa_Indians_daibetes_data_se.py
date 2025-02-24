# -*- coding: utf-8 -*-
"""
Blessed be your name Lord my God;
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

import pandas as pd
import numpy as np

from pathlib import Path

#-----LOAD THE DATA -------------------------------------------------   
df = pd.read_csv('diabetes.csv')
print(df)


column_names=df.axes[1] # df.columns 
feature_list=column_names[:-1] # remove the last column names which is 'outcome' 
X= df.loc[0 :, df.columns !='Outcome' ].values
y=df.loc[0 :, df.columns =='Outcome' ].values



# For a given feature in the list, here we are agoing to:
#---1) remove any duplicates;
#---2) sort the data and store in array cutoffs. 
#---3) Using each data point as a threshold, we calculate:
#------3.1) J score
#------3.3) Cutoff according to the highest J score
#------3.3) Plot the abvoe scores against x.

#Define functions to calculate J score and F score
def derive_j(tn, fp, fn, tp):
    tpr=tp/(tp+fn)
    fpr=fp/(tn+fp)
    #ppv=tp/(tp+fp)
    j=tpr-fpr
    #f=2*(ppv*tpr)/(ppv+tpr)
    return j


import matplotlib.pyplot as plt
J_max=[]
Cutoffs=[] #cutoff points according to the best threshold for each clinical fesature
for index in range(len(feature_list)):
    y_pred=X[:,index]
   
    #remove any duplicates
    from sklearn import metrics
    cutoffs=np.sort(list(set(y_pred)))
    J=[]
    for c in cutoffs:
         y_pred_binary = [1 if x >= c else 0 for x in y_pred]
         tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred_binary).ravel()
         j=derive_j(tn, fp, fn, tp) 
         J.append(j)
    
    # Now plot the line graphs
    plt.plot(cutoffs, J, color='orange', label='J score')
       
    j_max=np.max(J)
    J_max.append(j_max)    
    best_threshold_J=cutoffs[np.argmax(J)]
    Cutoffs.append(best_threshold_J)     
    plt.axvline(best_threshold_J,color = 'black', label = 'Best cutoff J',linestyle='dashed')
    print("Best threshold according to J : ",best_threshold_J )
    
    
    plt.xlabel(feature_list[index])
    plt.ylabel("J score")
    
    plt.legend()
    plt.show()


