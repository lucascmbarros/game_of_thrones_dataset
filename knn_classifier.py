# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:12:50 2019

@author: lucas.barros

Assignement 2: Game of Thrones predictions
"""


#################################
# Importing Libraries
#################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score # k-folds cross validation

'''
Results:
    Training Score: 0.8004
    Testing Score: 0.7634
    Cross-validation: 0.74
'''


#################################
# Importing file
#################################

file = 'got.xlsx'

df = pd.read_excel(file)


# Train, Test, Split
got_data = df.loc[: , ['isMarried',
                         'isNoble',
                         'out_popular',
                         'out_DOB',
                         'out_year',
                         'charnumber',
                         'book_3_4_5'                      
                           ]]


got_target =  df.loc[: , 'isAlive']



# Dividing the trainning and tests sets
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.25,
            random_state = 777,
            stratify = got_target)


# Tuning the model to get the best number of neighbors
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())

    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 13 neighbors is the most accurate - instantiate the model
knn_clf = KNeighborsClassifier(algorithm = 'auto',
                                   n_neighbors = 10)

# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

# Comparing the testing score to the training score.
print('Training Score:', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)


##################################################################
# Cross Validating the knn model with three folds
#################################################################
cv_knn_3 = cross_val_score(knn_clf,
                           got_data,
                           got_target,
                           cv = 3)


print(cv_knn_3)

print(pd.np.mean(cv_knn_3).round(3))


print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))


y_pred_train = knn_clf_fit.predict(X_train)
y_pred_test = knn_clf_fit.predict(X_test)

