# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:47:33 2020

@author: walke
"""

import matplotlib.style as style
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score, recall_score, \
     precision_score, confusion_matrix, plot_confusion_matrix, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from collections import defaultdict
from cleaning import cleaning, cleaning_w_impute
style.use('seaborn')
sns.set_style(style='darkgrid')
#%%
dc = pd.read_csv('dc-wikia-data.csv')
marvel = pd.read_csv('marvel-wikia-data.csv')
marvel = marvel.rename(columns={'Year': 'YEAR'})

data = pd.concat([dc,marvel])
data = cleaning(data)
#%%
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#%%
log = LogisticRegression(random_state=1, max_iter=1000,
                         multi_class='multinomial')

log.fit(X_train,y_train)

conf = plot_confusion_matrix(log,X_test,y_test,normalize='true',cmap='PuBu')
conf.ax_.grid(False)
conf.ax_.set_title('Logistic Regression Confusion Matrix')