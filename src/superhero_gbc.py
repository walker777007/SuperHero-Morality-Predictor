# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:01:18 2020

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
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
"""
#%%
"""
gbc = GradientBoostingClassifier()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
n_estimators = [100, 250, 500]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 4)]
max_depth = [5]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
learning_rate  = np.linspace(start = 0.025, stop = 0.1, num = 4)

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate,
               'random_state': [1]}

print(random_grid)

gb_random = RandomizedSearchCV(estimator = gbc, param_distributions = random_grid,
                               n_iter = 500, cv = 5, verbose=5, random_state=1,
                               n_jobs = -1)

gb_random.fit(X_train, y_train)

print("best parameters:", gb_random.best_params_)
"""
#%%
"""
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate = 0.1,
                                 max_depth=3, min_samples_split=2,
                                 min_samples_leaf = 1, 
                                 max_features=None, random_state=1)

gbc.fit(X_train,y_train)

conf = plot_confusion_matrix(gbc,X_test,y_test,normalize='true',cmap='PuBu')
conf.ax_.grid(False)
conf.ax_.set_title('Gradient Boosting Confusion Matrix')

fig, ax = plt.subplots()
ax.bar(np.asarray(data.drop(columns=['ALIGN']).columns),gbc.feature_importances_)
ax.set_xticklabels(data.drop(columns=['ALIGN']).columns,rotation=90)
ax.set_title('Gradient Boosting Feature Importance')
plt.tight_layout()
"""
#%%
"""
X = data.drop(columns=['ALIGN'])
y = data['ALIGN']
scores = defaultdict(list)

names = list(X.columns)
 
# cross validate the scores on a number of 
# different random splits of the data
splitter = ShuffleSplit(100, test_size=.3)

for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y.values[train_idx], y.values[test_idx]
    
    gbc.fit(X_train, y_train)
    
    acc = r2_score(y_test, gbc.predict(X_test))
    ## loop thru the features
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(y_test, gbc.predict(X_t))
        # relative accuracy decrease
        scores[names[i]].append((acc-shuff_acc)/acc)
        
score_series = pd.DataFrame(scores).mean()
scores = pd.DataFrame({'Mean Decrease Accuracy' : score_series})
scores.sort_values(by='Mean Decrease Accuracy').plot(kind='barh')
"""
#%%
"""
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate = 0.07,
                                 max_depth=4, min_samples_split=2,
                                 min_samples_leaf = 1, 
                                 max_features=None, random_state=1)

selector = RFECV(gbc, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
bad_features_gbc = ['Identity Unknown',
 'Known to Authorities Identity',
 'Amber Eyes',
 'Black Eyeballs',
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
 'Variable Eyes',
 'Violet Eyes',
 'Yellow Eyeballs',
 'Blue Hair',
 'Bronze Hair',
 'Gold Hair',
 'Magenta Hair',
 'Orange-brown Hair',
 'Pink Hair',
 'Platinum Blond Hair',
 'Purple Hair',
 'Silver Hair',
 'Strawberry Blond Hair',
 'Variable Hair',
 'Violet Hair',
 'Yellow Hair',
 'Genderfluid Characters',
 'Transgender Characters']
#%%
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate = 0.07,
                                 max_depth=4, min_samples_split=2,
                                 min_samples_leaf = 1, 
                                 max_features=None, random_state=1)

data = data.drop(columns=bad_features_gbc)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify = data['ALIGN'].values)

gbc.fit(X_train,y_train)

conf = plot_confusion_matrix(gbc,X_test,y_test,normalize='true',cmap='PuBu')
conf.ax_.grid(False)
conf.ax_.set_title('Gradient Boosting Confusion Matrix')

fig, ax = plt.subplots()
ax.bar(np.asarray(data.drop(columns=['ALIGN']).columns[np.argsort(gbc.feature_importances_)[::-1]]),
       gbc.feature_importances_[np.argsort(gbc.feature_importances_)[::-1]])
ax.set_xticklabels(data.drop(columns=['ALIGN']).columns[np.argsort(gbc.feature_importances_)[::-1]],
                   rotation=90)
ax.set_title('Gradient Boosting Feature Importance')
plt.tight_layout()
