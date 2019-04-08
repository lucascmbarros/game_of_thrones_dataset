# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:20:49 2019

@author: lucas.barros

"""

# Random Tree 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


file = 'got.xlsx'

df = pd.read_excel(file)

got_data = df.loc[: , ['isMarried',
                         'isNoble',
                         'out_popular',
                         'out_DOB',
                         'out_year',
                         'charnumber',
                         'book_3_4_5'                      
                           ]]

got_target =  df.loc[: , 'isAlive']


# Train, Test Split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = got_target)


# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 530,
                                     criterion = 'gini',
                                     max_depth = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     min_samples_leaf = 2,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 550,
                                     criterion = 'entropy',
                                     max_depth = 3,
                                     bootstrap = True,
                                     warm_start = False,
                                     min_samples_leaf = 3,
                                     random_state = 508)

# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)

# testing if the predictions are the same for both models
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()


# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)



#################################################################
# Cross Validating the tree model with three folds
#################################################################

# Cross Validating with gini
cv_treefull_3 = cross_val_score(full_forest_gini,
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



# Cross Validating with entropy
cv_treefull_3 = cross_val_score(full_forest_entropy,
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

y_pred_train = full_entropy_fit.predict(X_train)
y_pred_test =  full_entropy_fit.predict(X_test)

print('Training AUC Score',
      roc_auc_score(y_train, y_pred_train).round(4))

print('Testing AUC Score:',
      roc_auc_score(y_test, y_pred_test).round(4))

y_pred_train = full_gini_fit.predict(X_train)
y_pred_test =  full_gini_fit.predict(X_test)

print('Training AUC Score',
      roc_auc_score(y_train, y_pred_train).round(4))

print('Testing AUC Score:',
      roc_auc_score(y_test, y_pred_test).round(4))


#############################################################
# Tuning Hyperparameters with  RandomizedSearchCV
#############################################################

# Creating a hyperparameter grid
n_estimators = pd.np.arange(50, 1000, 100)
leaf_space = pd.np.arange(1, 4)
max_depth = pd.np.arange(1,4)
#bootstrap = [True, False]
#warm_start = [True, False]
criterion = ['gini', 'entropy']


param_grid = {'n_estimators' : n_estimators,
              'min_samples_leaf' : leaf_space,
              'max_depth': max_depth,
              'criterion' : criterion,              
              }

# Building the model object one more time
c_tree_3_hp = RandomForestClassifier(random_state = 777)



# Creating a RandomizedSearchCV object
c_tree_3_hp_cv = RandomizedSearchCV(c_tree_3_hp,
                                    param_grid,
                                    cv = 3,
                                    scoring = 'roc_auc'
                                   )


# Fit it to the training data
c_tree_3_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", 
      c_tree_3_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:",
      c_tree_3_hp_cv.best_score_.round(4))


###############################################################################
# Hyperparameter Tuning with GridSearchCV
###############################################################################


# Creating a hyperparameter grid
n_estimators = pd.np.arange(100, 1000)
leaf_space = pd.np.arange(1, 3)
max_depth = pd.np.arange(1, 5)
bootstrap = [True, False]
warm_start = [True, False]
criterion = ['gini', 'entropy']


param_grid = { 'max_depth' : max_depth,
              'min_samples_leaf' : leaf_space,
              'n_estimators' : n_estimators
              }

# Building the model object one more time
c_tree_hp = RandomForestClassifier(random_state = 777)

# Creating a GridSearchCV object
c_tree_hp_cv = GridSearchCV(c_tree_hp, param_grid, cv = 3)


# Fit it to the training data
c_tree_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:",
      c_tree_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:",
      c_tree_hp_cv.best_score_.round(4))





def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance_1.png')

########################
        
plot_feature_importances(full_gini_fit,
                         train = X_train,
                         export = False)



plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False)


