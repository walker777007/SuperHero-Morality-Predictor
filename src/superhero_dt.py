# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:07:55 2020

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
dt = DecisionTreeClassifier(random_state=1)

selector = RFECV(dt, step=1, cv=5, scoring='accuracy', verbose=5, n_jobs=-1)
selector = selector.fit(X_train, y_train)
bad_features = list(data.drop(columns=['ALIGN']).columns[~selector.support_])
"""
#%%
bad_features_dt = ['LGBT',
 'Identity Unknown',
 'Known to Authorities Identity',
 'No Dual Identity',
 'Amber Eyes',
 'Black Eyeballs',
 'Black Eyes',
 'Blue Eyes',
 'Compound Eyes',
 'Gold Eyes',
 'Green Eyes',
 'Grey Eyes',
 'Hazel Eyes',
 'Magenta Eyes',
 'Multiple Eyes',
 'No Eyes',
 'One Eye',
 'Orange Eyes',
 'Photocellular Eyes',
 'Pink Eyes',
 'Purple Eyes',
 'Red Eyes',
 'Silver Eyes',
 'Variable Eyes',
 'Violet Eyes',
 'White Eyes',
 'Yellow Eyeballs',
 'Yellow Eyes',
 'Auburn Hair',
 'Blue Hair',
 'Bronze Hair',
 'Gold Hair',
 'Green Hair',
 'Grey Hair',
 'Magenta Hair',
 'Orange Hair',
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
#%%
dt = DecisionTreeClassifier(random_state=1)

#data = data.drop(columns=bad_features_dt)
X = data.drop(columns=['ALIGN']).values
y = data['ALIGN'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dt.fit(X_train,y_train)

conf = plot_confusion_matrix(dt,X_test,y_test,normalize='true',cmap='YlGn')
conf.ax_.grid(False)
conf.ax_.set_title('Decision Tree Confusion Matrix')

fig, ax = plt.subplots()
ax.bar(np.asarray(data.drop(columns=['ALIGN']).columns[np.argsort(dt.feature_importances_)[::-1]]),
       dt.feature_importances_[np.argsort(dt.feature_importances_)[::-1]])
ax.set_xticklabels(data.drop(columns=['ALIGN']).columns[np.argsort(dt.feature_importances_)[::-1]],
                   rotation=90)
ax.set_title('Decision Tree Feature Importance')
plt.tight_layout()
