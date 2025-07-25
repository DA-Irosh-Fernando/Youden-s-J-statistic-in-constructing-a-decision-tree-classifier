# -*- coding: utf-8 -*-
"""
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

folder=Path("C:/Users/Me/Pima Indians Diabetes Dataset/")
df = pd.read_csv(folder/'diabetes.csv')



column_names=df.axes[1] # df.columns 
feature_list=column_names[:-1] # remove the last column names which is 'outcome' 
X= df.loc[0 :, df.columns !='Outcome' ].values
y=df.loc[0 :, df.columns =='Outcome' ].values



import JscoreBinariser

jbn=JscoreBinariser.JscoreBinariser()
indices,X_reduced=jbn.reduce_features(X,y,3)
X_reduced_bn=jbn.binarise(X_reduced, y)

#----------------SPLIT THE DATA FOR TRAINING AND TESTING------------------------------------------------------  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reduced_bn, y, test_size=0.5, random_state=16)



#----------------------------------ProbabilityTree3Classifier MODEL------------------------------------------------------

import ProbabilityTree3Classifier as pt

myTree=pt.ProbabilityTree3Classifier()
myTree.fit(X_train,y_train)
y_pred=myTree.predict(X_test)







from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay,roc_auc_score, precision_recall_curve, classification_report
cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Non-diabetic","Diabetic"])
print(classification_report(y_test, y_pred, ))

print("AUROC using binary : ",roc_auc_score(y_test,y_pred))

y_pred_proba=myTree.predict_proba(X_test)
print("AUROC using probablities : ",roc_auc_score(y_test,y_pred_proba))


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
print("J score: ", j)
print("Accuracy : ", accuracy)


#---------------------------------------PLOT AREA UNDER THE CURVE---------------------------------------------

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

#---------------------------------------WRITE TO A CSV FILE---------------------------------------------
import csv

with open('C:\\Users\\Me\\eHealth_paper_scores_JScore.csv','w',newline='') as csvfile:
    fieldnames =['Index','y_pred','y_test']
    writer=csv.DictWriter(csvfile, fieldnames =fieldnames)
    writer.writeheader()
    for i in range(len(y_pred_proba)):
        writer.writerow({'Index':i,'y_pred':y_pred_proba[i],'y_test':y_test[i]})

#---------------------------------------------------------------------------------------------------------------

