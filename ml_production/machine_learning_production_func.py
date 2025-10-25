#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning to Estimate Production Function for Cocoa Farms
- Compare ML methods
- Predict yield across number of shade trees using selected ML model

Created on Mon Mar 20 10:04:05 2023

@author: Jac
"""

# visualize the data


import os
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import Bounds

import multiprocessing as mp
from time import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

path = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()

#%%
# Load dataset
dataset = pd.read_csv('../../../02-Data/Interim/CMS/Sefwi Bekwai/farms_long_with_GEE_CMS.csv')
#dataset = pd.read_csv('../../../02-Data/Interim/CMS/farms_long_with_GEE_CMS_merged.csv')
# dataset = dataset.drop(['farm_id','farmerid','caseid',
#               'region','district', 'opsname','opsid','opscode','community',
#               'commu_size', 'commu_size_elig', 'land_community' ,
#               'oper_area','num_bag_beans_','num_bag_beans_pa','num_shade_tree_pa','num_shade_tree_',
#               'communityid','planting_m','total_income_pull_','land_size_acre_new_',
#               # 'hectares_1','farmsize_ha', 'total_plot_area'], axis=1)
dataset = dataset.dropna(axis = 1)
#dataset['num_bag_beans_cat'] = round(dataset['num_bag_beans_pa_w'],1).astype(str)
print(dataset.dtypes)

encoder = OneHotEncoder()
encoded_results = encoder.fit_transform(dataset.drop(['plotid'],axis=1).select_dtypes(include=['object'])).toarray()
numeric_results = dataset.drop(['num_bag_beans_pa_w'],axis=1).select_dtypes(include=['int','float'])
print(list(numeric_results))
array = np.concatenate((numeric_results, encoded_results), axis=1)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('cocoa_type_cat').size())

#%%
# data visualization
dataset[dataset.columns[-5:-1]].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset[dataset.columns[-5:-1]].hist()
plt.show()
# scatter
scatter_matrix(dataset[dataset.columns[-5:-1]])
plt.show()


#%%
# Split-out validation dataset
X = numeric_results.values
y = dataset['num_bag_beans_pa_w']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Standardize variables
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

#%%
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis())) # less prone to overfitting
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB())) # assumes feature independence
#models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []

for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
#plt.show()
plt.savefig('../../../04-Output/Baseline/algorithm_comparison_baseline2024.pdf')

#%%
# Make predictions on validation dataset
model = LinearDiscriminantAnalysis()
model.fit(X_train_scaled, Y_train)
predictions = model.predict(X_validation)
std = np.sqrt(np.sum((predictions - Y_validation)**2)/np.size(predictions))

predictions2 = model.predict(X)
std2 = np.sqrt(np.sum((predictions2 - y)**2)/np.size(predictions2))
print(std)
print(std2)

#%%
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#%%
# Get the feature coefficients from the LDA model
coefficients = model.coef_
# Compute the absolute value of the coefficients
abs_coefficients = np.abs(coefficients)

# Compute the total absolute value of the coefficients for each feature
total_abs_coefficients = np.sum(abs_coefficients, axis=0)

# Sort the features by importance
indices = np.argsort(total_abs_coefficients)[::-1]

# Print the feature ranking
print("Feature ranking:")

for i, coef in enumerate(model.coef_[0][:73]):
    print("%s: %f" % (numeric_results.columns[i], abs(coef)))




#%% Prediction
# select LDA
# df_predict = pd.read_csv('../../../02-Data/Interim/CMS/farms_long_to_predict.csv')
df_predict = pd.read_csv('../../../02-Data/Interim/CMS/Sefwi Bekwai/farms_long_to_predict.csv')
# df_predict = df_predict.drop(['farm_id','farmerid','caseid',
#               'region','district', 'opsname','opsid','opscode','community',
#               'commu_size', 'commu_size_elig', 'land_community' ,
#               'oper_area','num_bag_beans_','num_bag_beans_pa','num_shade_tree_pa','num_shade_tree_',
#               'communityid','planting_m','total_income_pull_','land_size_acre_new_',
#               'hectares_1','farmsize_ha', 'total_plot_area', 'citrus', '_merge','uniform','dup'], axis=1)
df_predict = df_predict.dropna(axis = 1)
print(df_predict.dtypes)

encoder = OneHotEncoder()
encoded_results_predict = encoder.fit_transform(df_predict.drop(['plotid'],axis=1).select_dtypes(include=['object'])).toarray()
numeric_results_predict = df_predict.drop(['num_bag_beans_pa_w'],axis=1).select_dtypes(include=['int','float'])
print(numeric_results_predict.shape)
# X_preload = np.concatenate((numeric_results_predict, encoded_results_predict), axis=1)
X_predict = numeric_results_predict.values
# Standardize variables
scaler = preprocessing.StandardScaler().fit(X_predict)
X_predict_scaled = scaler.transform(X_predict)
# Predict
predictions_preload = model.predict(X_predict_scaled)

#%%
df_predict['y_predict'] = predictions_preload

#pd.DataFrame(df_predict).to_csv("../../../02-Data/Interim/CMS/yield_predicted.csv")
pd.DataFrame(df_predict).to_csv("../../../02-Data/Interim/CMS/Sefwi Bekwai/yield_predicted.csv")
