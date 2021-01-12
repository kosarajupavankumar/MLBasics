# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 08:54:03 2021

@author: pavankumar.kosaraju
"""
# importing a required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import pickle 

#importing a dataset
dataset = pd.read_excel('dataGYM.xlsx')
df=dataset.copy()

del df['BMI']
del df['Class']

labelencoder =LabelEncoder()
df['Prediction']= labelencoder.fit_transform(df['Prediction'])

# checking the null values 
nullvalues=dataset.isnull().sum()

#Training the model 
X= df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model_GYM = RandomForestClassifier(n_estimators=20)

model_GYM.fit(X_train,y_train)

print(model_GYM)

# make the predictions 
expected = y_test
predicted = model_GYM.predict(X_test)


# summarize the fit of the model 
metrics.classification_report((expected), predicted)
metrics.confusion_matrix(expected, predicted)

# saving the model to disk 

pickle.dump(model_GYM, open('model.pkl','wb'))

# Loading the model to compare the results 
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[40,5.6,70]]))
