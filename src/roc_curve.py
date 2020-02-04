# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:41:50 2020

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
from superhero_gbc import gbc, X_test, y_test
X_test_gbc = X_test
y_test_gbc = y_test
from superhero_rfc import rfc, X_test, y_test
X_test_rfc = X_test
y_test_rfc = y_test
from superhero_log import log, X_test, y_test
X_test_log = X_test
y_test_log = y_test
from superhero_dt import dt, X_test, y_test
X_test_dt = X_test
y_test_dt = y_test
from superhero_svc import svm, X_test, y_test
X_test_svm = X_test
y_test_svm = y_test
from superhero_knn import knn, X_test, y_test
X_test_knn = X_test
y_test_knn = y_test
#%%
fig, ax = plt.subplots()
plot_roc_curve(gbc,X_test_gbc,y_test_gbc, ax=ax)
plot_roc_curve(rfc,X_test_rfc,y_test_rfc, ax=ax)
plot_roc_curve(log,X_test_log,y_test_log, ax=ax)
plot_roc_curve(dt,X_test_dt,y_test_dt, ax=ax)
plot_roc_curve(svm,X_test_svm,y_test_svm, ax=ax)
plot_roc_curve(knn,X_test_knn,y_test_knn, ax=ax)
plt.plot(np.linspace(0,1,11),np.linspace(0,1,11),color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

