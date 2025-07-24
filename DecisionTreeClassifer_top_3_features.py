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



#----------------SPLIT THE DATA FOR TRAINING AND TESTING------------------------------------------------------  


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.5, random_state=16)




from sklearn.tree import DecisionTreeClassifier

feature_names=feature_list
target_names=['Diabetic','Non-diabetic']
#tree_clf=DecisionTreeClassifier(criterion='gini',max_depth=None, random_state=42)
#tree_clf=DecisionTreeClassifier(criterion='gini',max_depth=2, random_state=42)
#tree_clf=DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=42)
#tree_clf=DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=42)
tree_clf=DecisionTreeClassifier(criterion='log_loss',max_depth=3, random_state=42)
#tree_clf=DecisionTreeClassifier(criterion='entropy',max_depth=None, random_state=42)
#tree_clf=DecisionTreeClassifier(criterion='log_loss',max_depth=None, random_state=42)

tree_clf.fit(X_train,y_train)
tree_clf.score(X_train,y_train)
y_pred=tree_clf.predict(X_test)
y_pred_prob=tree_clf.predict_proba(X_test)



#---------------------------------------WRITE THE OUTCOME TO A CSV FILE---------------------------------------------
import csv

with open('C:\\Users\\Me\\eHealth_paper_scores_Log_loss_w3.csv','w',newline='') as csvfile:
    fieldnames =['Index','y_pred','y_test']
    writer=csv.DictWriter(csvfile, fieldnames =fieldnames)
    writer.writeheader()
    for i in range(len(y_pred_prob)):
        x=y_pred_prob[i]
        y=np.array(y_test[i])
        writer.writerow({'Index':i,'y_pred':x[1],'y_test':y[0]})

#---------------------------------------------------------------------------------------------------------------



from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay,roc_auc_score, precision_recall_curve, classification_report
cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Non-diabetic","Diabetic"])
disp.plot()
print(classification_report(y_test, y_pred, ))

print("AUROC binary : ",roc_auc_score(y_test,y_pred))

y_pred_proba=tree_clf.predict_proba(X_test)
print("AUROC probablities : ",roc_auc_score(y_test,y_pred_proba[:,1]))

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
        y_pred_proba[:,1],
        name="Diabetic vs Non=diabetic",
        color="darkorange",
        plot_chance_level=True,
    )
_ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve using ProbabilityTree3Classifier",
    )

#---------------------------------------------------------------------------------------------------------------

from matplotlib import pyplot as plt
from sklearn import tree
tree.plot_tree(tree_clf, feature_names=feature_list[indices],proportion=True)
plt.show()

