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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score, recall_score, accuracy_score,\
     precision_score, confusion_matrix, plot_confusion_matrix, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV,\
     RandomizedSearchCV, ShuffleSplit
from collections import defaultdict
from cleaning import cleaning
import warnings
warnings.filterwarnings('ignore')
style.use('seaborn')
sns.set_style(style='darkgrid')
#%%
dc = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/dc-wikia-data-edited.csv')
marvel = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/marvel-wikia-data-edited.csv')
marvel = marvel.rename(columns={'Year': 'YEAR'})

data = pd.concat([dc,marvel])
data = cleaning(data)
#%%
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#%%
log = LogisticRegression(random_state=1, max_iter=1000, C=0.9)
"""
selector = RFECV(log, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
bad_features_log = ['APPEARANCES', 'YEAR', 'Black Eyes', 'Yellow Eyeballs', 'Orange-brown Hair']

data = data.drop(columns=bad_features_log)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#%%
log = LogisticRegression(random_state=1, max_iter=1000, C=0.9)

cv_results = cross_validate(log, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

log.fit(X_train,y_train)

log_accuracy = 0.6498549125443802
log_recall = 0.766295364259065
log_precision = 0.6596375382830825
#%%
"""
conf = plot_confusion_matrix(log,X_test,y_test,normalize='true',cmap='YlOrBr')
conf.ax_.grid(False)
conf.ax_.set_title('Logistic Regression Confusion Matrix')
"""