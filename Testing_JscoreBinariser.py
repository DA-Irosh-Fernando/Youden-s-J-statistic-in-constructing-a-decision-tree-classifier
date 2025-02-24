# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:03:08 2025

@author: daipf
"""

import pandas as pd
import numpy as np

#Create a sample data set
X=np.array([[10,12,13,15,11,9,13,11],
            [10,31,20,21,5,18,20,6],
            [42,41,34,81,55,48,60,56]])
y=[0,1,1,1,0,1,0,0]
size=X.shape[0]
index=list(range(0,X.shape[1]))
print(X)
print(index)
print("---------------------------------------------------------------------")
X=np.transpose(X)
print(X)

print("---------------------------------------------------------------------")
feature_list=np.array(['Feature_1','Feature_2','Feature_3'])
df=pd.DataFrame(X,index=index, columns=feature_list)
df['Target']=y

print(df)
print("---------------------------------------------------------------------")
#import JscoreBinariser as jbn
import JscoreBinariser

jbn=JscoreBinariser.JscoreBinariser()

#Testing cutoffs
#---------------
cutoffs, J_max =jbn.derive_cutoffs(X,y)
for i in range (X.shape[1]):
    print (feature_list[i], " cutoff : ", cutoffs[i], " Maximum J score : ", J_max[i])


print("---------------------------------------------------------------------")

#Testing feature reduction
#------------------------------------------------
# reduced the size of the feature set from 3 to 2

reduced_size=2 
indices,X_reduced=jbn.reduce_features(X,y,reduced_size)
print("Reduced feature set : ",feature_list[indices])

print("---------------------------------------------------------------------")
#Need an np.array for this
J_max=np.array(jbn.J_max)
reduced_feature_list=feature_list[indices]
J_max_for_reduced_feature_list=J_max[indices]
for i in range (X_reduced.shape[1]):
    print (reduced_feature_list[i], " Maximum J score : ", J_max_for_reduced_feature_list[i])

print("---------------------------------------------------------------------")
#Testying binarisation
#---------------------
#Binarise the reduced data set 
X_reduced_bn=jbn.binarise(X_reduced, y)
print(X_reduced)



