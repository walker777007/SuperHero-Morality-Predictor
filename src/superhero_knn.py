# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:34:52 2020

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
bad_features_knn = ['Identity Unknown',
 'Compound Eyes',
 'Bronze Hair',
 'Transgender Characters']
data = data.drop(columns=bad_features_knn)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#%%
knn = KNeighborsClassifier(n_neighbors=25, p=1)

cv_results = cross_validate(knn, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

knn.fit(X_train,y_train)

knn_accuracy = 0.6753440410261242
knn_recall = 0.7876587188653721
knn_precision = 0.6798322584733862
#%%
"""
conf = plot_confusion_matrix(knn,X_test,y_test,normalize='true',cmap='OrRd')
conf.ax_.grid(False)
conf.ax_.set_title('K Nearest Neighbors Confusion Matrix')
"""