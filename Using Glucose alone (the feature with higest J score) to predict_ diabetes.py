# -*- coding: utf-8 -*-
"""
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

#-----Load the data -------------------------------------------------   
df = pd.read_csv('diabetes.csv')
print(df)

#data  = df.loc[1:44 :, df1.columns == "Unnamed: 0"].values    

column_names=df.axes[1] # df.columns 
feature_list=column_names[:-1] # remove the last column names which is 'outcome' 
X= df.loc[0 :, df.columns !='Outcome' ].values

i=df.columns.get_loc("Glucose")
glucose=X[:,i]
#y_pred= df.loc[0 :, df.columns =='Glucose' ].values
y=df.loc[0 :, df.columns =='Outcome' ].values


#-----Calculate J scores and find the highest which is used as the best threshoshold
#-----to binarise data 
# Here is what we are agoing to:
#---1) remove any duplicates;
#---2) sort the data and store in array cutoffs. 
#---3) Using each data point as a threshold, we calculate:
#------3.1) J score
#------3.3) Cutoff according to the highest J score
#------3.3) Plot the abobe scores against x.


#Define functions to calculate J score and F score
def derive_j(tn, fp, fn, tp):
    tpr=tp/(tp+fn)
    fpr=fp/(tn+fp)
    #ppv=tp/(tp+fp)
    j=tpr-fpr
    #f=2*(ppv*tpr)/(ppv+tpr)
    return j



import matplotlib.pyplot as plt

#remove any duplicates
from sklearn import metrics
cutoffs=np.sort(list(set(glucose)))

J=[]
 
for c in cutoffs:
     glucose_binary = [1 if x >= c else 0 for x in glucose]
     tn, fp, fn, tp = metrics.confusion_matrix(y, glucose_binary).ravel()
     j=derive_j(tn, fp, fn, tp) 
     J.append(j)

# Now plot the line graphs
plt.plot(cutoffs, J, color='orange', label='J score')
   
j_max=np.max(J)
best_threshold_J=cutoffs[np.argmax(J)]
plt.axvline(best_threshold_J,color = 'black', label = 'Best cutoff J',linestyle='dashed')
print("Best threshold according to J : ",best_threshold_J )

plt.xlabel("Glucose")
plt.ylabel("J score")

plt.legend()
plt.show()



#---------------Birarise using the best threshold (the glucose level corresponding to the maximum J score) as the cutoff----------

glucose_binary_best= [1 if x >=best_threshold_J else 0 for x in glucose]


# No need to split the data for training and testing since there is no model to train.

y_pred_binary=glucose_binary_best
y_pred_cont=glucose

from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay,roc_auc_score, precision_recall_curve, classification_report
cm=confusion_matrix(y, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Non-diabetic","Diabetic"])
print(classification_report(y, y_pred_binary, ))

print("AUROC binary : ",roc_auc_score(y,y_pred_binary))
print("AUROC Continous data : ",roc_auc_score(y,y_pred_cont))


from sklearn import metrics
tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred_binary).ravel()
tpr=tp/(tp+fn)
fpr=fp/(tn+fp)
ppv=tp/(tp+fp)
npv=tn/(tn+fn)
tnr=tn/(fp+tn)
j=tpr-fpr
accuracy=(tp+tn)/(tn+fp+fn+tp)

print("TP : ", tp)
print("TN : ", tn)
print("FP : ", fp)
print("FN : ", fn)
print("TPR : ", tpr)
print("PPV : ", ppv)
print("TNR : ", tnr)
print("NPV : ", npv)
print("J : ", j)
print("Accuracy : ", accuracy)


#----------------Plot the area under the curve with the binarised data---------------------------------------------

from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(
        y,
        y_pred_binary,
        name="Diabetic vs Non=diabetic",
        color="darkorange",
        plot_chance_level=True,
    )
_ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve using Glucose (binary)",
    )

from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
print(classification_report(y, y_pred_binary, ))



#---------------Plot the area under the curve with the original (non-binarised) data-----------------------

from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(
        y,
        y_pred_cont,
        name="Diabetic vs Non=diabetic",
        color="darkorange",
        plot_chance_level=True,
    )
_ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve using Gulcose (continous)",
    )





