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
colors = sns.color_palette()
style.use('seaborn')
sns.set_style(style='darkgrid')
#%%
def plot_roc(X, y, clf, color, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=color, label=type(clf).__name__ +' (AUC = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
#%%
from superhero_gbc import gbc, X_train, y_train
X_train_gbc = X_train
y_train_gbc = y_train
from superhero_rfc import rfc, X_train, y_train
X_train_rfc = X_train
y_train_rfc = y_train
from superhero_log import log, X_train, y_train
X_train_log = X_train
y_train_log = y_train
from superhero_dt import dt, X_train, y_train
X_train_dt = X_train
y_train_dt = y_train
from superhero_svc import svm, X_train, y_train
X_train_svm = X_train
y_train_svm = y_train
from superhero_knn import knn, X_train, y_train
X_train_knn = X_train
y_train_knn = y_train
from superhero_abc import abc, X_train, y_train
X_train_abc = X_train
y_train_abc = y_train
from superhero_xgb import xgb, X_train, y_train
X_train_xgb = X_train
y_train_xgb = y_train
#%%
plot_roc(X_train_xgb,y_train_xgb,xgb, colors[0])
plot_roc(X_train_gbc,y_train_gbc,gbc, colors[1])
plot_roc(X_train_rfc,y_train_rfc,rfc, colors[2])
plot_roc(X_train_abc,y_train_abc,abc, colors[3])
plot_roc(X_train_knn,y_train_knn,knn, colors[4])
plot_roc(X_train_svm,y_train_svm,svm, colors[5])
plot_roc(X_train_log,y_train_log,log, colors[6])
plot_roc(X_train_dt,y_train_dt,dt, colors[7])
plt.plot(np.linspace(0,1,11),np.linspace(0,1,11),color='k',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('5 CV Mean ROC Curves')
plt.tight_layout()
#plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/ROC.png', dpi=640)
#plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/ROC_slides.png', dpi=640)
