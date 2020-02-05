# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:05:55 2020

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

abc = AdaBoostClassifier(n_estimators=500, learning_rate = 0.07, random_state=1)
"""
selector = RFECV(abc, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
bad_features_abc = ['Identity Unknown',
 'Known to Authorities Identity',
 'No Dual Identity',
 'Amber Eyes',
 'Black Eyeballs',
 'Black Eyes',
 'Compound Eyes',
 'Gold Eyes',
 'Grey Eyes',
 'Hazel Eyes',
 'Magenta Eyes',
 'Multiple Eyes',
 'No Eyes',
 'One Eye',
 'Orange Eyes',
 'Pink Eyes',
 'Purple Eyes',
 'Silver Eyes',
 'Unknown Eyes',
 'Variable Eyes',
 'Violet Eyes',
 'White Eyes',
 'Yellow Eyeballs',
 'Auburn Hair',
 'Bronze Hair',
 'Brown Hair',
 'Gold Hair',
 'Green Hair',
 'Grey Hair',
 'Magenta Hair',
 'Orange-brown Hair',
 'Pink Hair',
 'Platinum Blond Hair',
 'Purple Hair',
 'Red Hair',
 'Silver Hair',
 'Strawberry Blond Hair',
 'Variable Hair',
 'Violet Hair',
 'White Hair',
 'Yellow Hair',
 'Agender Characters',
 'Genderfluid Characters',
 'Male Characters',
 'Transgender Characters']

data = data.drop(columns=bad_features_abc)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#%%
abc = AdaBoostClassifier(n_estimators=500, learning_rate = 0.07, random_state=1)

#cv_results = cross_validate(abc, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

abc.fit(X_train,y_train)

abc_accuracy = 0.6770527179777083
abc_recall =  0.7908695474197585
abc_precision = 0.6806630693420754
#%%
"""
conf = plot_confusion_matrix(xgb,X_test,y_test,normalize='true',cmap='GnBu')
conf.ax_.grid(False)
conf.ax_.set_title('XGBoost Confusion Matrix')
"""