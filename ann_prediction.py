# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:45:16 2019

@author: acostalago@gmail.com
"""

""" Importing all the libraries and the dataset"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

""" Over-sampling using SMOTE """

# Oversampling of the data due to imbalance in the dependent variable
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X )
os_data_y= pd.DataFrame(data=os_data_y)


""" Design of the ANN """
with open("best3.pkl", "rb") as f:   # Unpickling
    parameters = pickle.load(f)

classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(activation="relu", units=parameters['units'], kernel_initializer="uniform", input_dim=10))
classifier.add(Dropout(rate = 0.1, seed = 0))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=parameters['units'], kernel_initializer="uniform"))
classifier.add(Dropout(rate = 0.1, seed = 0))

# Adding the third hidden layer
#classifier.add(Dense(activation="relu", units=parameters['units'], kernel_initializer="uniform"))
#classifier.add(Dropout(rate = 0.1, seed = 0))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(os_data_X, os_data_y, batch_size = parameters['batch_size'], epochs = parameters['epochs'])

""" Cross Validation and Prediction """
# Predicting the Test set results
y_pred = classifier.predict(X_test)
predictions = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, predictions)
accu = (cm[0][0]+cm[1][1])/len(predictions)

# Receiver operating characteristic curve  (ROC curve)
logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right")


def build_classifier():
    classifier = Sequential()
    # Input layer
    classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform", input_dim=10))
    classifier.add(Dropout(rate = 0.1, seed = 0))
    # Second hidden layer
    classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))    
    classifier.add(Dropout(rate = 0.1, seed = 0))
    # Third hidden layer
#    classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))    
#    classifier.add(Dropout(rate = 0.1, seed = 0))
    # Output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier, batch_size = parameters['batch_size'], epochs = parameters['epochs'])
accuracies = cross_val_score(estimator = classifier, X = os_data_X, y = os_data_y, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()