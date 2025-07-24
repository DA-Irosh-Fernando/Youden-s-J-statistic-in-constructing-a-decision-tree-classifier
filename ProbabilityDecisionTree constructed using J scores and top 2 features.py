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


#-----Load the data set ---------------------------------------------------------------------------------   
df = pd.read_csv('diabetes.csv')
print(df)

column_names=df.axes[1] # df.columns 
feature_list=column_names[:-1] # remove the last column names which is 'outcome' 
X= df.loc[0 :, df.columns !='Outcome' ].values
y=df.loc[0 :, df.columns =='Outcome' ].values


#----------------------------------Reduce the featureset and binarise using the maximum J scores-------------------


#import JscoreBinariser as jbn
import JscoreBinariser

jbn=JscoreBinariser.JscoreBinariser()
reduced_size=2
indices,X_reduced=jbn.reduce_features(X,y,reduced_size)
X_reduced_bn=jbn.binarise(X_reduced, y)


#----------- split X and y into training and testing sets----------------------------------------------------------

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.5, random_state=16)
X_train, X_test, y_train, y_test = train_test_split(X_reduced_bn, y, test_size=0.5, random_state=16)




#----------------------------------ProbabilityTree2ClassifierL------------------------------------------------------

import ProbabilityTree2Classifier as pt

myTree=pt.ProbabilityTree2Classifier()
myTree.fit(X_train,y_train)
y_pred=myTree.predict(X_test)




#----------------------------------Derive the evaluation metrics------------------------------------------------------

from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay,roc_auc_score, precision_recall_curve, classification_report
cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Non-diabetic","Diabetic"])
print(classification_report(y_test, y_pred, ))

print("AUROC binary : ",roc_auc_score(y_test,y_pred))

y_pred_proba=myTree.predict_proba(X_test)
print("AUROC probablities : ",roc_auc_score(y_test,y_pred_proba))


from sklearn import metrics
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
tpr=tp/(tp+fn)
fpr=fp/(tn+fp)
ppv=tp/(tp+fp)
npv=tn/(tn+fn)
tnr=tn/(fp+tn)
f=2*(ppv*tpr)/(ppv+tpr)
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
print("F score: ", f)
print("J : ", j)
print("Accuracy : ", accuracy)


#---------------------------------------PLOT AREA UNDER THE CURVE---------------------------------------------
#---Use predicted binary outcome values for the ROC curve
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(
        y_test,
        y_pred,
        name="Diabetic vs Non=diabetic",
        color="darkorange",
        plot_chance_level=True,
    )
_ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve using ProbabilityTree3Classifier",
    )
#---Use predicted prbability values for the ROC curve
from sklearn.metrics import RocCurveDisplay
display = RocCurveDisplay.from_predictions(
        y_test,
        y_pred_proba,
        name="Diabetic vs Non=diabetic",
        color="darkorange",
        plot_chance_level=True,
    )
_ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve using ProbabilityTree3Classifier",
    )


