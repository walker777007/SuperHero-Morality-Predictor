# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:54:23 2020

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score, recall_score, \
     precision_score, confusion_matrix, plot_confusion_matrix, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
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
"""
svm = SVC(random_state=1,gamma='auto')

selector = RFECV(svm, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
svm = SVC(random_state=1, gamma='auto')

X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

svm.fit(X_train,y_train)

conf = plot_confusion_matrix(svm,X_test,y_test,normalize='true',cmap='RdPu')
conf.ax_.grid(False)
conf.ax_.set_title('Support Vector Machine Confusion Matrix')
