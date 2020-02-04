# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:37:27 2020

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
data = data.drop(columns=['page_id','urlslug'])
data = data.rename(columns={'GSM':'LGBT'})
data['LGBT'] = data['LGBT'].map({np.nan:0})
data['LGBT'] = data['LGBT'].map({np.nan:1, 0:0})
data['ALIGN'] = data['ALIGN'].map({'Good Characters': 0, 'Bad Characters':1})
#%%
bad_males = np.sum((np.array(data['ALIGN']==1) & np.array(data['SEX']=='Male Characters')))
good_males = np.sum((np.array(data['ALIGN']==0) & np.array(data['SEX']=='Male Characters')))
bad_females = np.sum((np.array(data['ALIGN']==1) & np.array(data['SEX']=='Female Characters')))
good_females = np.sum((np.array(data['ALIGN']==0) & np.array(data['SEX']=='Female Characters')))

bad_male_pct = 100*bad_males/(bad_males+good_males)
good_male_pct = 100*good_males/(bad_males+good_males)
bad_female_pct = 100*bad_females/(bad_females+good_females)
good_female_pct = 100*good_females/(bad_females+good_females)

fig, ax = plt.subplots()
ax.bar([0, 1], [bad_male_pct,bad_female_pct], color='red')
ax.bar([0, 1], [good_male_pct,good_female_pct], bottom=[bad_male_pct,bad_female_pct], color='blue')
ax.set_title('Morality by Sex')
ax.set_xticks([0,1])
ax.set_xticklabels(['Male','Female'])
ax.set_xlabel('Sex')
ax.set_ylabel('Percentage')
plt.tight_layout()
plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/Morality_by_Sex.png', dpi=640)

"""fig, ax = plt.subplots()
ax.bar(['Male','Female'],
       [100*data['ALIGN'][data['SEX']=='Male Characters'].sum()/data['ALIGN'][~data['ALIGN'].isnull()].sum(),100*data['ALIGN'][data['SEX']=='Female Characters'].sum()/data['ALIGN'][~data['ALIGN'].isnull()].sum()],
       color='red')
ax.set_title('Bad Characters by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Percentage')
plt.tight_layout()"""
#%%
eyes = np.unique(data['EYE'][~data['EYE'].isnull()])
bad_eyes=[]
good_eyes=[]
for eye in eyes:
    bad_eyes.append((data['ALIGN'][data['EYE']==eye]==1).sum())
    good_eyes.append((data['ALIGN'][data['EYE']==eye]==0).sum())
    
eyes = eyes[np.add(bad_eyes,good_eyes)>50]
top_bad_eyes = np.array(bad_eyes)[np.add(bad_eyes,good_eyes)>50]
top_good_eyes = np.array(good_eyes)[np.add(bad_eyes,good_eyes)>50]

fig, ax = plt.subplots()
bars = ax.bar(eyes[np.argsort(np.divide(top_bad_eyes,np.add(top_bad_eyes,top_good_eyes)))[::-1]],
       100*np.divide(top_bad_eyes,np.add(top_bad_eyes,top_good_eyes))[np.argsort(np.divide(top_bad_eyes,np.add(top_bad_eyes,top_good_eyes)))[::-1]])
i=0
for bar in bars:
    if i<7:
        bar.set_color('red')
    if i>6:
        bar.set_color('blue')
    i+=1
ax.set_title('Eye Color Morality')
ax.set_xlabel('Eye Color')
ax.set_ylabel('Percentage')
plt.tight_layout()
plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/Eye_Color_Morality.png', dpi=640)
#%%
hairs = np.unique(data['HAIR'][~data['HAIR'].isnull()])
bad_hairs=[]
good_hairs=[]
for hair in hairs:
    bad_hairs.append((data['ALIGN'][data['HAIR']==hair]==1).sum())
    good_hairs.append((data['ALIGN'][data['HAIR']==hair]==0).sum())
    
hairs = hairs[np.add(bad_hairs,good_hairs)>60]
top_bad_hairs = np.array(bad_hairs)[np.add(bad_hairs,good_hairs)>60]
top_good_hairs = np.array(good_hairs)[np.add(bad_hairs,good_hairs)>60]

fig, ax = plt.subplots()
bars = ax.bar(hairs[np.argsort(np.divide(top_bad_hairs,np.add(top_bad_hairs,top_good_hairs)))[::-1]],
       100*np.divide(top_bad_hairs,np.add(top_bad_hairs,top_good_hairs))[np.argsort(np.divide(top_bad_hairs,np.add(top_bad_hairs,top_good_hairs)))[::-1]])
i=0
for bar in bars:
    if i<7:
        bar.set_color('red')
    if i>6:
        bar.set_color('blue')
    i+=1
ax.set_title('Hair Color Morality')
ax.set_xlabel('Hair Color')
ax.set_ylabel('Percentage')
plt.tight_layout()
plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/Hair_Color_Morality.png', dpi=640)
#%%
bad_x = np.arange(len(data['APPEARANCES'][data['ALIGN']==1][~np.isnan(data['APPEARANCES'][data['ALIGN']==1])]))
good_x = np.arange(len(data['APPEARANCES'][data['ALIGN']==0][~np.isnan(data['APPEARANCES'][data['ALIGN']==0])]))

bad_appearances = np.array(data['APPEARANCES'][data['ALIGN']==1][~np.isnan(data['APPEARANCES'][data['ALIGN']==1])])
good_appearances = np.array(data['APPEARANCES'][data['ALIGN']==0][~np.isnan(data['APPEARANCES'][data['ALIGN']==0])])

fig, ax = plt.subplots()
ax.fill_between(good_x, good_appearances[np.argsort(good_appearances)[::-1]],
                 np.zeros(good_x.shape[0]),color='blue')
ax.fill_between(bad_x, bad_appearances[np.argsort(bad_appearances)[::-1]],
                 np.zeros(bad_x.shape[0]),color='red')
ax.set_xlim(-50,5000)
ax.set_title('Appearances by Morality')
ax.set_xlabel('Top 250 Characters')
ax.set_ylabel('Appearances')
plt.tight_layout()
plt.savefig('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/plots/Appearances_by_Morality.png', dpi=640)
