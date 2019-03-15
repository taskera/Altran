# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:30:11 2019

@author: acostalago@gmail.com
"""

""" Importing all the libraries and the dataset"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('InputData.csv')
X = dataset.loc[:, ['age','lifestyle','zip code', 'family status', 'car', 'sports', 'earnings', 'Living area']].values
y = dataset.loc[:, ['label']].values

""" Data exploration """

# Analysis of the dependent variable -> Imbalance
dataset.groupby('label').mean()
dataset.groupby(['label','lifestyle']).count()
dataset.groupby(['label','family status']).count()
dataset.groupby(['label','car']).count()
dataset.groupby(['label','sports']).count()
dataset.groupby(['label','Living area']).count()

#Plots of the imbalance dependant variable
dataset['label'].value_counts()
x = ['Response', 'No response']
count_no_res = len(dataset[dataset['label']=='no response'])
count_res = len(dataset[dataset['label']=='response'])
lif = [count_res, count_no_res]
plt.subplots(figsize=(8, 8))
plt.bar(x, lif, label = 'Response')
plt.title('Response Frequency', fontsize=18)
plt.xlabel('Label', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()
pct_of_no_res = count_no_res/(count_no_res+count_res)
print("percentage of no response is", pct_of_no_res*100)
pct_of_res = count_res/(count_no_res+count_res)
print("percentage of response", pct_of_res*100)

# Plot of the age for each labels
x = ['15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70']
out = pd.cut(dataset.age, bins=list(range(15, 80, 5)), labels = x, include_lowest=True)
liv_tot = out.value_counts().sort_index()
liv_res = out[dataset.label=='response'].value_counts()
liv_res = 100*liv_res.sort_index().values/liv_tot.values
liv_nores = out[dataset.label=='no response'].value_counts()
liv_nores = 100*liv_nores.sort_index().values/liv_tot.values
plt.subplots(figsize=(8, 8))
plt.bar(x, liv_res, label = 'Response')
plt.bar(x, liv_nores, bottom=liv_res, label = 'No response')
plt.title('Response vs Age', fontsize=18)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()


# Plot of the earnings for each labels
x = ['20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '100000', '110000', '120000', '130000', '140000']
out = pd.cut(dataset.earnings, bins=list(range(20000, 160000,10000)), labels = x, include_lowest=True)
liv_tot = out.value_counts().sort_index()
liv_res = out[dataset.label=='response'].value_counts()
liv_res = 100*liv_res.sort_index().values/liv_tot.values
liv_nores = out[dataset.label=='no response'].value_counts()
liv_nores = 100*liv_nores.sort_index().values/liv_tot.values
plt.subplots(figsize=(8, 8))
plt.bar(x, liv_res, label = 'Response')
plt.bar(x, liv_nores, bottom=liv_res, label = 'No response')
plt.title('Response vs Earnings', fontsize=18)
plt.xlabel('Earnings', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()

# Plot of the Living area for each label
x = ['Urban', 'Rural']
urban_response = 0
urban_noresponse = 0
rural_response = 0
rural_noresponse = 0
for i,j in enumerate(dataset['Living area']):
    if j == 'urban' and y[i]=='response':
        urban_response += 1
    elif j == 'urban' and y[i]=='no response':
        urban_noresponse += 1
    elif j == 'rural' and y[i]=='response':
        rural_response += 1
    else:
        rural_noresponse += 1
urban_response_100 = 100*urban_response/(urban_response+urban_noresponse)
urban_noresponse_100 = 100*urban_noresponse/(urban_response+urban_noresponse)
rural_response_100 = 100*rural_response/(rural_response+rural_noresponse)
rural_noresponse_100 = 100*rural_noresponse/(rural_response+rural_noresponse)
liv = [[urban_response_100,rural_response_100],[urban_noresponse_100,rural_noresponse_100]]

plt.subplots(figsize=(8, 8))
plt.bar(x, liv[0], label = 'Response')
plt.bar(x, liv[1], bottom=liv[0], label = 'No response')
plt.title('Response vs Living area', fontsize=18)
plt.xlabel('Living area', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()

# Plot of the lifestyle for each labels
x = ['Active', 'Cozily', 'Healthy']
active_response = 0
active_noresponse = 0
cozily_response = 0
cozily_noresponse = 0
healthy_response = 0
healthy_noresponse = 0
for i,j in enumerate(dataset['lifestyle']):
    if j == 'active' and y[i]=='response':
        active_response += 1
    elif j == 'active' and y[i]=='no response':
        active_noresponse += 1
    elif j == 'cozily' and y[i]=='response':
        cozily_response += 1
    elif j == 'cozily' and y[i]=='no response':
        cozily_noresponse += 1
    elif j == 'healthy' and y[i]=='response':
        healthy_response += 1
    else:
        healthy_noresponse += 1
active_response_100 = 100*active_response/(active_response+active_noresponse)
active_noresponse_100 = 100*active_noresponse/(active_response+active_noresponse)
cozily_response_100 = 100*cozily_response/(cozily_response+cozily_noresponse)
cozily_noresponse_100 = 100*cozily_noresponse/(cozily_response+cozily_noresponse)
healthy_response_100 = 100*healthy_response/(healthy_response+healthy_noresponse)
healthy_noresponse_100 = 100*healthy_noresponse/(healthy_response+healthy_noresponse)
lif = [[active_response_100,cozily_response_100, healthy_response_100],[active_noresponse_100,cozily_noresponse_100, healthy_noresponse_100]]

plt.subplots(figsize=(8, 8))
plt.bar(x, lif[0], label = 'Response')
plt.bar(x, lif[1], bottom=lif[0], label = 'No response')
plt.title('Response vs Lifestyle', fontsize=18)
plt.xlabel('Lifestyle', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()

# Plot of the family status for each labels
x = ['Married', 'Single']
married_response = 0
married_noresponse = 0
single_response = 0
single_noresponse = 0
for i,j in enumerate(dataset['family status']):
    if j == 'married' and y[i]=='response':
        married_response += 1
    elif j == 'married' and y[i]=='no response':
        married_noresponse += 1
    elif j == 'single' and y[i]=='response':
        single_response += 1
    else:
        single_noresponse += 1
married_response_100 = 100*married_response/(married_response+married_noresponse)
married_noresponse_100 = 100*married_noresponse/(married_response+married_noresponse)
single_response_100 = 100*single_response/(single_response+single_noresponse)
single_noresponse_100 = 100*single_noresponse/(single_response+single_noresponse)
liv = [[married_response_100,single_response_100],[married_noresponse_100,single_noresponse_100]]

plt.subplots(figsize=(8, 8))
plt.bar(x, liv[0], label = 'Response')
plt.bar(x, liv[1], bottom=liv[0], label = 'No response')
plt.title('Response vs Family status', fontsize=18)
plt.xlabel('Family status', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()

# Plot of the car type for each labels
x = ['Practical', 'Expensive']
car_response = 0
car_noresponse = 0
car_exp_response = 0
car_exp_noresponse = 0
for i,j in enumerate(dataset['car']):
    if j == 'practical' and y[i]=='response':
        car_response += 1
    elif j == 'practical' and y[i]=='no response':
        car_noresponse += 1
    elif j == 'expensive' and y[i]=='response':
        car_exp_response += 1
    else:
        car_exp_noresponse += 1
car_response_100 = 100*car_response/(car_response+car_noresponse)
car_noresponse_100 = 100*car_noresponse/(car_response+car_noresponse)
car_exp_response_100 = 100*car_exp_response/(car_exp_response+car_exp_noresponse)
car_exp_noresponse_100 = 100*car_exp_noresponse/(car_exp_response+car_exp_noresponse)
liv = [[car_response_100,car_exp_response_100],[car_noresponse_100,car_exp_noresponse_100]]

plt.subplots(figsize=(8, 8))
plt.bar(x, liv[0], label = 'Response')
plt.bar(x, liv[1], bottom=liv[0], label = 'No response')
plt.title('Response vs Car type', fontsize=18)
plt.xlabel('Car type', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()

# Plot of the sport for each labels
x = ['Athletics', 'Soccer', 'Badminton', 'None']
Athletics_response = 0
Athletics_noresponse = 0
Soccer_response = 0
Soccer_noresponse = 0
Badminton_response = 0
Badminton_noresponse = 0
None_response = 0
None_noresponse = 0
for i,j in enumerate(dataset['sports']):
    if j == 'athletics' and y[i]=='response':
        Athletics_response += 1
    elif j == 'athletics' and y[i]=='no response':
        Athletics_noresponse += 1
    elif j == 'soccer' and y[i]=='response':
        Soccer_response += 1
    elif j == 'soccer' and y[i]=='no response':
        Soccer_noresponse += 1
    elif j == 'badminton' and y[i]=='response':
        Badminton_response += 1
    elif j == 'badminton' and y[i]=='no response':
        Badminton_noresponse += 1
    elif pd.isnull(j) and y[i]=='response':
        None_response += 1
    else:
        None_noresponse += 1
        
Athletics_response_100 = 100*Athletics_response/(Athletics_response+Athletics_noresponse)
Athletics_noresponse_100 = 100*Athletics_noresponse/(Athletics_response+Athletics_noresponse)
Soccer_response_100 = 100*Soccer_response/(Soccer_response+Soccer_noresponse)
Soccer_noresponse_100 = 100*Soccer_noresponse/(Soccer_response+Soccer_noresponse)
Badminton_response_100 = 100*Badminton_response/(Badminton_response+Badminton_noresponse)
Badminton_noresponse_100 = 100*Badminton_noresponse/(Badminton_response+Badminton_noresponse)
None_response_100 = 100*None_response/(None_response+None_noresponse)
None_noresponse_100 = 100*None_noresponse/(None_response+None_noresponse)
lif = [[Athletics_response_100,Soccer_response_100, Badminton_response_100, None_response_100],[Athletics_noresponse_100,Soccer_noresponse_100, Badminton_noresponse_100, None_noresponse_100]]

plt.subplots(figsize=(8, 8))
plt.bar(x, lif[0], label = 'Response')
plt.bar(x, lif[1], bottom=lif[0], label = 'No response')
plt.title('Response vs Sport', fontsize=18)
plt.xlabel('Sport', fontsize=16)
plt.ylabel('Percentage of Response', fontsize=16)
plt.legend()
plt.show()


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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


""" Over-sampling using SMOTE """

# Oversampling of the data due to imbalance in the dependent variable
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X )
os_data_y= pd.DataFrame(data=os_data_y)

print("length of oversampled data is ",len(os_data_X))
print("Number of response in oversampled data",os_data_y[os_data_y==1].count())
print("Number of no response",os_data_y[os_data_y==0].count())
print("Proportion of response data in oversampled data is ",os_data_y[os_data_y==1].count()/len(os_data_X))
print("Proportion of no response data in oversampled data is ",os_data_y[os_data_y==0].count()/len(os_data_X))

# Check that the data is now balanced
os_data_y.mean()


""" Recursive Feature Elimination """
# Check if all independent variables are related to the dependent variable
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
# Since all are related (True) no variable will be eliminated

""" Creating the logistic regression model"""
# Statistical model of the logistic regression to provide table with the results
logit_model=sm.Logit(os_data_y,os_data_X)
result=logit_model.fit()
print(result.summary2())

# Logistic regression model
logreg = LogisticRegression()
logreg.fit(os_data_X, os_data_y)

# Prediction with the test data splitted previously
predictions = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

# Receiver operating characteristic curve (ROC curve)
logit_roc_auc = roc_auc_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right")