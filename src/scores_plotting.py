# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:55:08 2020

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
colors = sns.color_palette()
style.use('seaborn')
sns.set_style(style='darkgrid')
#%%
from superhero_gbc import gbc, gbc_accuracy, gbc_recall, gbc_precision
from superhero_rfc import rfc, rfc_accuracy, rfc_recall, rfc_precision
from superhero_log import log, log_accuracy, log_recall, log_precision
from superhero_dt import dt, dt_accuracy, dt_recall, dt_precision
from superhero_svc import svm, svm_accuracy, svm_recall, svm_precision
from superhero_knn import knn, knn_accuracy, knn_recall, knn_precision
from superhero_abc import abc, abc_accuracy, abc_recall, abc_precision
from superhero_xgb import xgb, xgb_accuracy, xgb_recall, xgb_precision
#%%
barWidth = 0.1
 
# set height of bar
xgb_bars = 100*np.array([xgb_accuracy, xgb_precision, xgb_recall])
gbc_bars = 100*np.array([gbc_accuracy, gbc_precision, gbc_recall])
rfc_bars = 100*np.array([rfc_accuracy, rfc_precision, rfc_recall])
abc_bars = 100*np.array([abc_accuracy, abc_precision, abc_recall])
knn_bars = 100*np.array([knn_accuracy, knn_precision, knn_recall])
svm_bars = 100*np.array([svm_accuracy, svm_precision, svm_recall])
log_bars = 100*np.array([log_accuracy, log_precision, log_recall])
dt_bars = 100*np.array([dt_accuracy, dt_precision, dt_recall])

# Set position of bar on X axis
r1 = np.arange(len(xgb_bars))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
r8 = [x + barWidth for x in r7]
 
# Make the plot
plt.bar(r1, xgb_bars, color=colors[0], width=barWidth, edgecolor='white', label=type(xgb).__name__)
plt.bar(r2, gbc_bars, color=colors[1], width=barWidth, edgecolor='white', label=type(gbc).__name__)
plt.bar(r3, rfc_bars, color=colors[2], width=barWidth, edgecolor='white', label=type(rfc).__name__)
plt.bar(r4, abc_bars, color=colors[3], width=barWidth, edgecolor='white', label=type(abc).__name__)
plt.bar(r5, knn_bars, color=colors[4], width=barWidth, edgecolor='white', label=type(knn).__name__)
plt.bar(r6, svm_bars, color=colors[5], width=barWidth, edgecolor='white', label=type(svm).__name__)
plt.bar(r7, log_bars, color=colors[6], width=barWidth, edgecolor='white', label=type(log).__name__)
plt.bar(r8, dt_bars, color=colors[7], width=barWidth, edgecolor='white', label=type(dt).__name__)

plt.xticks([r + barWidth for r in range(len(xgb_bars))], ['Accuracy', 'Precision', 'Recall'])
plt.title('Model Scores')
plt.xlabel('Models')
plt.ylabel('Percentage')
plt.legend(loc=2)
plt.ylim(0,100)
plt.tight_layout()
#plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/Model_Scores.png', dpi=640)
