# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:02:13 2020

@author: KUNAL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataset=pd.read_csv(r"C:\Users\KUNAL\Documents\aiml\ML project\nasa.csv",encoding='latin-1')


dataset.drop( ['Close Approach Date'], axis = 1, inplace = True)
dataset.drop( ['Orbiting Body'], axis = 1, inplace = True)
dataset.drop( ['Orbit Determination Date'], axis = 1, inplace = True)
dataset.drop( ['Equinox'], axis = 1, inplace = True)
dataset.drop(dataset.iloc[:,[0,1,2,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]],axis=1, inplace = True)
dataset.drop(dataset.iloc[:,[1,3]] , axis = 1, inplace = True)
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#Dropping the Redundant and outliner columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=22)

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
import seaborn as sns
sns.set(color_codes=True)

dataframe_training = pd.DataFrame()
dataframe_training['Est Dia in KM(min)'] = X_train['Est Dia in KM(min)']
dataframe_training['Miss Dist.(Astronomical)'] = y_train
ax = sns.regplot(x="Est Dia in KM(min)", y="Miss Dist.(Astronomical)", data= dataframe_training)

dataframe_test = pd.DataFrame()
dataframe_test['Est Dia in KM(min)'] = X_test['Est Dia in KM(min)']
dataframe_test['Miss Dist.(Astronomical)'] = y_test
ax = sns.regplot(x="Est Dia in KM(min)", y="Miss Dist.(Astronomical)", data= dataframe_training)

print('Coefficients: \n', regressor.coef_)
print('Intercept: \n',regressor.intercept_)
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print("Variance score: {}".format(r2_score(y_test, y_pred)))

residual_test = np.column_stack([y_test,y_pred])
residual_test = pd.DataFrame(residual_test)
print(residual_test)
residual_test.columns='Y_test predictions'.split()
print(residual_test.columns)
sns.jointplot(x='Y_test', y='predictions', data=residual_test, kind='reg')

stats.levene(residual_test['Y_test'], residual_test['predictions'])



