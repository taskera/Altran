# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:24:16 2019

@author: acostalago@gmail.com
"""

""" Load the dataset """

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.preprocessing import StandardScaler


# Importing the dataset
dataset = pd.read_csv('InputData.csv')
X = dataset.loc[:, ['age','lifestyle','zip code', 'family status', 'car', 'sports', 'earnings', 'Living area']].values
y = dataset.loc[:, ['label']].values


""" Encoding categorical data """
# Encoding of the inputs
# Encoding of lifestyle (column 2) 1--> cozily, 0-->active, 2--> healthy
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])


# Encoding of family status (column 4) 0--> married, 1-->single
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

# Encoding of car (column 5) 1-->practical, 0-->expensive
labelencoder_X_3 = LabelEncoder()
X[:, 4] = labelencoder_X_3.fit_transform(X[:, 4])

## Taking care of missing data --> adding to new category
X[pd.isnull(X[:, 5]), 5] = 'none'
# Encoding of sports (column 6) 0-->athletics, 3-->soccer, 1--> badminton, 2-->none
labelencoder_X_4 = LabelEncoder()
X[:, 5] = labelencoder_X_4.fit_transform(X[:, 5])

# Encoding of living area (column 8) 1-->urban, 0-->rural
labelencoder_X_5 = LabelEncoder()
X[:, 7] = labelencoder_X_5.fit_transform(X[:, 7])

# Encode categorical data with more than 2 cases 
# 1 0--> cozily, 0 0-->active, 0 1--> healthy
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# 1 1-->athletics, 0 1-->soccer, 0 0--> badminton, 1 0-->none (One more column was added previously!)
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
# eliminate unnecesary columns
X[X[:,0]==1,2] = 1
X[X[:,0]==1,3] = 1
X = X[:, 2:]

""" New Column order: 
sport | sport | lifestyle | lifestyle | age | zip code | family status | car | earnings | living area
"""

# Encoding the output 0 -> no response, 1-> response
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
# Remove the mean and scale to unit variance
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Create network into function for parameter search and cross validation
def build_classifier(units, optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", units=units, kernel_initializer="uniform", input_dim=10))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(activation="relu", units=units, kernel_initializer="uniform"))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))    
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Parameters for the search of optimal architecture
parameters = {'batch_size': [32, 48, 96],
              'epochs': [500, 700, 1000],
              'optimizer': ['adam', 'rmsprop'],
              'units': [6, 8, 10, 12]}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best = grid_search.best_params_
best['accuracy'] = grid_search.best_score_

# Save parameters for later use in the ANN prediction
with open('best3.pkl','wb') as f:
    pickle.dump(best,f)
