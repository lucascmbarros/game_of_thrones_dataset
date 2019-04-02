# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:26:49 2019

@author: lucas.barros

Assignment 2: Game of Thrones predictions
"""

###############################################################################
# Creating the models for Predictions
###############################################################################

#############################
# Importing Libraries
#############################

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.metrics import roc_auc_score    # AUC value
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

#####################
# Importing file
#####################

file = 'got.xlsx'
df = pd.read_excel(file)

##############################################################################
# Train, Test, Split 
##############################################################################

got_data_1 = df.loc[: , ['isMarried',
                         'isNoble',
                         'out_popular',
                         'out_DOB',
                         'out_year',
                         'charnumber',
                         'book_3_4_5'                      
                           ]]

got_target_1 =  df.loc[: , 'isAlive']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
            got_data_1,
            got_target_1.values.ravel(),
            test_size = 0.10,
            random_state = 508,
            stratify = got_target_1)


##############################################################################
# Instatiating Gradient Boosting Classifier 
##############################################################################

gbm_3_1 = GradientBoostingClassifier(loss = 'deviance',
                                          learning_rate = 0.86,
                                          n_estimators = 150,
                                          max_depth =1,
                                          min_samples_leaf = 1,
                                          criterion = 'friedman_mse',
                                          warm_start = False,
                                          random_state = 508
                                          )


gbm_basic_fit_1 = gbm_3_1.fit(X_train_1, y_train_1)

gbm_basic_predict_1 = gbm_basic_fit_1.predict(X_test_1)

# Training and Testing Scores
print('Training Score', gbm_basic_fit_1.score(X_train_1, y_train_1).round(4))
print('Testing Score:', gbm_basic_fit_1.score(X_test_1, y_test_1).round(4))

'''
The model is returning a predictive score of 85%. Train Score and Test Score 
indicates that the data is a bit underfit, but nothing that will influence so
much on the model. 
'''

gbm_basic_train_1 = gbm_basic_fit_1.score(X_train_1, y_train_1)
gmb_basic_test_1  = gbm_basic_fit_1.score(X_test_1, y_test_1)


##############################################################################
# Cross Validating the tree model with three folds 
##############################################################################
cv_treefull_3_1 = cross_val_score(gbm_3_1,
                           got_data_1,
                           got_target_1,
                           cv = 3,
                           scoring = 'roc_auc')


print(cv_treefull_3_1)

cv_auc = pd.np.mean(cv_treefull_3_1).round(3)
print(pd.np.mean(cv_treefull_3_1).round(3))


print('\nAverage: ',
      pd.np.mean(cv_treefull_3_1).round(3),
      '\nMinimum: ',
      min(cv_treefull_3_1).round(3),
      '\nMaximum: ',
      max(cv_treefull_3_1).round(3))

'''
My ROC Cross_validation indicates that 81.40% of the times my testing score 
was right when predicting with cross-validated sample data. 
'''

##############################################################################
# Checking the AUC value 
##############################################################################
y_pred_train_1 = gbm_basic_fit_1.predict(X_train_1)
y_pred_test_1 =  gbm_basic_fit_1.predict(X_test_1)

print('Training AUC Score',
      roc_auc_score(y_train_1, y_pred_train_1).round(4))

print('Testing AUC Score:',
      roc_auc_score(y_test_1, y_pred_test_1).round(4))

roc_auc = roc_auc_score(y_test_1, y_pred_test_1).round(4)



# Saving best model scores
model_scores_df = pd.DataFrame({'GBM_Test_score': [gmb_basic_test_1],
                                'GBM_CV_AUC': [cv_auc],
                                'AUC_Test_Score' :[roc_auc]})


model_scores_df.to_excel("Model_Results.xlsx")



# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test_1,
                                     'GBM_Predicted': gbm_basic_predict_1
                                     })


model_predictions_df.to_excel("Model_Predictions.xlsx")


'''
Results:
    
    According to this results, we can see two sets of variables being 
    determinant upon the character survival. We can call as 'non-actionable'
    variables (variables that the character has no power to control) the:
    Popularity Outliers, the present data from from date of birth(the data
    without the missing values), which year the 
    character is in (out_year), in which point he was mentioned in the novel
    (charnumber), and if he/she was mentioned in book 3, 4, and/or 5. It is 
    difficult to advice what a character should or not do in order to survive
    according to these intrinsic characteristics of him/her. 
    
    Nevertheless, using the Logistic Regression to check which variables were
    influencing positively and negatively towards being alive or not, we see 
    that being popular, having a present date of birth in the dataset and 
    being introduced later in the novel will influence negatively on the pro-
    bability of being alive.
    
    Doing a general analysis of the why of this, we can suppose that these 
    characteristics are from characters that have main roles in the novel, hence
    being present in the main wars, conflicts, disputes, arguments... These
    events are likely putting the character in moments of danger, and according
    to the novel and data, people die the most in the north (Night's Watch, Red
    Wedding, conflicts in Winterfell, Brave Companions...) of Westeros.
    
    That said, the main recommendations would be to stay in the south, probably
    near Dorne. Being married and noble influences positively on the probability
    of being alive - the best (if the character is both) is to make a deal
    with the House Martell do live under their protection or simply their
    territory.

'''

'''
##############################################################################
# Tuning the HyperParameters
##############################################################################

leaf_space = pd.np.arange(1, 3)
warm_start = [True, False]
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 1000, 50)
depth_space = pd.np.arange(1, 4)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}



# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = RandomizedSearchCV(gbm_grid, 
                                 param_grid,
                                 cv = 3,
                                 scoring = 'roc_auc')



# Fit it to the training data
gbm_grid_cv.fit(X_train_2, y_train_2)



# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

'''


'''
##############################################################################
# Logistic Regression
##############################################################################
logistic_sig= smf.logit(formula = """isAlive ~ isNoble +
                                              out_popular +
                                             out_DOB +
                                             out_year +
                                             charnumber+
                                             book_3_4_5""",
                                              data = df)


results_logistic_sig = logistic_sig.fit()


print(results_logistic_sig.summary())
'''




