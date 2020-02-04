# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:44:57 2020

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
"""
rfc = RandomForestClassifier()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 5)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state': [1]}

print(random_grid)

rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid,
                               n_iter = 500, cv = 5, verbose=5, random_state=1,
                               n_jobs = -1)

rf_random.fit(X_train, y_train)

print("best parameters:", rf_random.best_params_)
"""
#%%
rfc = RandomForestClassifier(n_estimators=300,min_samples_split=6,
                             min_samples_leaf=1,max_features=None,
                             max_depth=10,bootstrap=True,random_state=1)
"""
selector = RFECV(rfc, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
bad_features_rfc = ['Identity Unknown',
 'Known to Authorities Identity',
 'Black Eyeballs',
 'Compound Eyes',
 'Magenta Eyes',
 'No Eyes',
 'One Eye',
 'Silver Eyes',
 'Violet Eyes',
 'Yellow Eyeballs',
 'Bronze Hair',
 'Gold Hair',
 'Magenta Hair',
 'Orange-brown Hair',
 'Pink Hair',
 'Platinum Blond Hair',
 'Silver Hair',
 'Violet Hair',
 'Genderfluid Characters',
 'Transgender Characters']

data = data.drop(columns=bad_features_rfc)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#%%
rfc = RandomForestClassifier(n_estimators=300,min_samples_split=6,
                             min_samples_leaf=1,max_features=None,
                             max_depth=10,bootstrap=True,random_state=1)

cv_results = cross_validate(rfc, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

rfc.fit(X_train,y_train)

dt_accuracy = 0.6923614753774101
dt_recall = 0.7773279091798665
dt_precision = 0.7020127020067484
#%%
"""
conf = plot_confusion_matrix(rfc,X_test,y_test,normalize='true',cmap='Greens')
conf.ax_.grid(False)
conf.ax_.set_title('Random Forest Confusion Matrix')

fig, ax = plt.subplots()
ax.bar(np.asarray(data.drop(columns=['ALIGN']).columns[np.argsort(rfc.feature_importances_)[::-1]]),
       rfc.feature_importances_[np.argsort(rfc.feature_importances_)[::-1]])
ax.set_xticklabels(data.drop(columns=['ALIGN']).columns[np.argsort(rfc.feature_importances_)[::-1]],
                   rotation=90)
ax.set_title('Random Forest Feature Importance')
plt.tight_layout()
"""