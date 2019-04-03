# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:23:17 2019

@author: lucas.barros

Assignement 2: Game of Thrones predictions
"""

#################################
# Importing Libraries
#################################
from sklearn.tree import DecisionTreeClassifier # Regression trees
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation
from sklearn.metrics import roc_auc_score    # AUC value
from sklearn.model_selection import GridSearchCV

'''
    Results:
        Training Score 0.8539
        Testing Score: 0.823
        Cross-validation Score: 0.834
'''

#################################
# Importing file
#################################

file = 'got.xlsx'

df = pd.read_excel(file)


# Train, Test, Split
got_data = df.loc[: , ['out_DOB',
                       'charnumber',
                       'book_3_4_5',
                        'out_year',
                        'out_popular'
                        ]]

got_target =  df.loc[: , 'isAlive']

# This is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.25,
            random_state = 777,
            stratify = got_target)

# Instatiating
tree_full = DecisionTreeClassifier(criterion = 'entropy',
                                   max_depth = 7,
                                   random_state = 777,
                                   min_samples_leaf = 2)

# Fitting
tree_full_fit = tree_full.fit(X_train, y_train)

# Scoring

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))

# 0.8459
# 0.8109



##################################################################
# Cross Validating the tree model with three folds
#################################################################
cv_treefull_3 = cross_val_score(tree_full,
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


###############################################################################
# Hyperparameter Tuning with GridSearchCV
###############################################################################


########################
# Optimizing for one hyperparameter
########################


# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 20)
criterion = ['entropy', 'gini']
splitter = ['best', 'random']



param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion,
              'splitter' : splitter
              }

# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 777)



# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)



# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", 
      c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:",
      c_tree_2_hp_cv.best_score_.round(4))


#################################################################
# Checking the AUC value
#################################################################
y_pred_train = tree_full_fit.predict(X_train)
y_pred_test =  tree_full_fit.predict(X_test)

print('Training AUC Score',
      roc_auc_score(y_train, y_pred_train).round(4))

print('Testing AUC Score:',
      roc_auc_score(y_test, y_pred_test).round(4))

###################################
# Feature importance function
##################################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Featur")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

########################
plot_feature_importances(tree_full_fit,
                         train = X_train,
                         export = True)



