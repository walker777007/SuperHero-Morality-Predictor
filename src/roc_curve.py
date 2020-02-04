# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:41:50 2020

@author: walke
"""
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
def plot_roc(X, y, clf_class, plot_name, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('17_' + plot_name + '.png')
    #plt.close()
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

