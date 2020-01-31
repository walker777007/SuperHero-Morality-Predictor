# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:48:37 2020

@author: walke
"""

import pandas as pd 
import numpy as np
import scipy.stats as stats

def cleaning(data):
    np.random.seed(1)
    data = data.reset_index().drop(columns=['index'])
    data['HAIR'][data['EYE'] == 'Auburn Hair'] = data['HAIR'][data['EYE'] == 'Auburn Hair'].fillna('Auburn Hair')
    data['EYE'][data['EYE'] == 'Auburn Hair'] = np.nan
    data = data.rename(columns={'GSM':'LGBT'})
    data['LGBT'] = data['LGBT'].map({np.nan:0})
    data['LGBT'] = data['LGBT'].map({np.nan:1, 0:0})
    #data['ALIGN'][data['ALIGN']=='Reformed Criminals'] = 'Neutral Characters'
    data = data[data['ALIGN'] != 'Neutral Characters']
    data = data[data['ALIGN'] != 'Reformed Criminals']
    data = data[pd.notnull(data['ALIGN'])]
    data = data[data['name'] != 'GenderTest']
    data['ALIVE'] = data['ALIVE'].map({'Deceased Characters':0, 'Living Characters':1})
    data = data.drop(columns=['page_id','name','urlslug','FIRST APPEARANCE'])
    
    data = data.dropna()
    data = data.join(pd.get_dummies(data['ID']))
    data = data.drop(columns=['ID'])
    data = data.join(pd.get_dummies(data['EYE']))
    data = data.drop(columns=['EYE'])
    data = data.join(pd.get_dummies(data['HAIR']))
    data = data.drop(columns=['HAIR'])
    data = data.join(pd.get_dummies(data['SEX']))
    data = data.drop(columns=['SEX'])
    #data = data.drop(columns=['Identity Unknown','Black Eyeballs','Platinum Blond Hair','Genderfluid Characters'])
    data['ALIGN'] = data['ALIGN'].map({'Good Characters': 1, 'Bad Characters':0})
    return data

def cleaning_w_impute(data):
    np.random.seed(1)
    data = data.reset_index().drop(columns=['index'])
    data['HAIR'][data['EYE'] == 'Auburn Hair'] = data['HAIR'][data['EYE'] == 'Auburn Hair'].fillna('Auburn Hair')
    data['EYE'][data['EYE'] == 'Auburn Hair'] = np.nan
    data = data.rename(columns={'GSM':'LGBT'})
    data['LGBT'] = data['LGBT'].map({np.nan:0})
    data['LGBT'] = data['LGBT'].map({np.nan:1, 0:0})
    #data['ALIGN'][data['ALIGN']=='Reformed Criminals'] = 'Neutral Characters'
    data = data[data['ALIGN'] != 'Neutral Characters']
    data = data[data['ALIGN'] != 'Reformed Criminals']
    data = data[pd.notnull(data['ALIGN'])]
    data = data[data['name'] != 'GenderTest']
    data['ALIVE'] = data['ALIVE'].map({'Deceased Characters':0, 'Living Characters':1})
    data = data.drop(columns=['page_id','name','urlslug','FIRST APPEARANCE'])
    
    data['YEAR'] = data['YEAR'].fillna(np.mean(data['YEAR']))
    data['APPEARANCES'] = data['APPEARANCES'].fillna(np.mean(data['APPEARANCES']))
    data['ALIVE'] = data['ALIVE'].fillna(stats.mode(data['ALIVE'])[0][0])
    sex_probs = data['SEX'].value_counts()/data['SEX'].value_counts().sum()
    data['SEX'] = data['SEX'].fillna(np.random.choice(sex_probs.index,p=sex_probs.values))
    hair_probs = data['HAIR'].value_counts()/data['HAIR'].value_counts().sum()
    data['HAIR'] = data['HAIR'].fillna(np.random.choice(hair_probs.index,p=hair_probs.values))   
    eye_probs = data['EYE'].value_counts()/data['EYE'].value_counts().sum()
    data['EYE'] = data['EYE'].fillna(np.random.choice(eye_probs.index,p=eye_probs.values))     
    id_probs = data['ID'].value_counts()/data['ID'].value_counts().sum()
    data['ID'] = data['ID'].fillna(np.random.choice(id_probs.index,p=id_probs.values))

    data = data.join(pd.get_dummies(data['ID']))
    data = data.drop(columns=['ID'])
    data = data.join(pd.get_dummies(data['EYE']))
    data = data.drop(columns=['EYE'])
    data = data.join(pd.get_dummies(data['HAIR']))
    data = data.drop(columns=['HAIR'])
    data = data.join(pd.get_dummies(data['SEX']))
    data = data.drop(columns=['SEX'])
    data = data.drop(columns=['Identity Unknown','Compound Eyes','Platinum Blond Hair','Transgender Characters'])
    data['ALIGN'] = data['ALIGN'].map({'Good Characters': 1, 'Bad Characters':0})
    return data