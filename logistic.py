# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:57:57 2019

@author: lucas
"""

# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # logistic regression
from sklearn.metrics import roc_auc_score    # AUC value
from sklearn.metrics import confusion_matrix # confusion matrix
import seaborn as sns                        # visualizing the confusion matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

file = 'got.xlsx'

df = pd.read_excel(file)

'''
    Results:
        Training Score 0.8422
        Testing Score: 0.8107
        Cross-Validation Score: 0.811
'''



######################################
# Logistic regression
######################################


# Cheking for p-values
logistic_sig= smf.logit(formula = """isAlive ~ 
                                                isNoble +
                                              out_popular +
                                              out_DOB +
                                             out_year +
                                             charnumber+
                                             book_3_4_5""",
                                              data = df)


results_logistic_sig = logistic_sig.fit()


print(results_logistic_sig.summary())


results_logistic_sig.pvalues


# Other important summary statistics
print('AIC:', results_logistic_sig.aic.round(2))
print('BIC:', results_logistic_sig.bic.round(2))




# Train, Test, Split
got_data = df.loc[: , ['out_DOB',
                       'charnumber',
                       'book_3_4_5',
                        'out_year',
                        'out_popular'
                        ]]

got_target =  df.loc[: , 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.25,
            random_state = 777,
            stratify = got_target)


# Instantiate the model
logreg = LogisticRegression(solver = 'lbfgs',
                            C = 0.1)


logreg_fit = logreg.fit(X_train, y_train)


# Predictions
logreg_pred = logreg_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))


##################################################################
# Cross Validating the tree model with three folds
#################################################################
cv_treefull_3 = cross_val_score(logreg,
                                got_data,
                                got_target,
                                cv = 3)


print(cv_treefull_3)


print(pd.np.mean(cv_treefull_3).round(3))


print('\nAverage: ',
      pd.np.mean(cv_treefull_3).round(3),
      '\nMinimum: ',
      min(cv_treefull_3).round(3),
      '\nMaximum: ',
      max(cv_treefull_3).round(3))


#################################################################
# Checking the AUC value
#################################################################
y_pred_train = logreg_fit.predict(X_train)
y_pred_test =  logreg_fit.predict(X_test)

print('Training AUC Score',
      roc_auc_score(y_train, y_pred_train).round(4))

print('Testing AUC Score:',
      roc_auc_score(y_test, y_pred_test).round(4))



########################
# Confusion Matrix
########################

# Run the following code to create a confusion matrix
print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred))


# Visualizing the confusion matrix
labels = ['Not High Price', 'High Price']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Blues')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()
