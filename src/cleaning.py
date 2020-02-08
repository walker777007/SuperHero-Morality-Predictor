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
    data['HAIR'][data['HAIR']=='Light Brown Hair'] = 'Brown Hair'
    data['HAIR'][data['HAIR']=='Reddish Brown Hair'] = 'Auburn Hair'
    data['HAIR'][data['HAIR']=='Reddish Blond Hair'] = 'Strawberry Blond Hair'
    data['SEX'][data['SEX']=='Genderless Characters'] = 'Agender Characters'
    data['HAIR'][data['HAIR']=='Bald'] = 'No Hair'
    data['ID'] = data['ID'].fillna('No Dual Identity')
    data['SEX'] = data['SEX'].fillna('Agender Characters')
    data['HAIR'] = data['HAIR'].fillna('No Hair')
    data['EYE'] = data['EYE'].fillna('Unknown Eyes')
    data = data.rename(columns={'GSM':'LGBT'})
    data['LGBT'] = data['LGBT'].map({np.nan:0})
    data['LGBT'] = data['LGBT'].map({np.nan:1, 0:0})
    #data['ALIGN'][data['ALIGN']=='Reformed Criminals'] = 'Neutral Characters'
    data = data[data['ALIGN'] != 'Neutral Characters']
    data = data[data['ALIGN'] != 'Reformed Criminals']
    data = data[pd.notnull(data['ALIGN'])]
    data = data[data['name'] != 'GenderTest']
    data['ALIVE'] = data['ALIVE'].map({'Deceased Characters':0, 'Living Characters':1})
    data.set_index('name', inplace = True)
    data = data.drop(columns=['page_id','urlslug','FIRST APPEARANCE'])
    
    data['YEAR'] = data['YEAR'].fillna(np.nanmedian(data['YEAR']))
    data['APPEARANCES'] = data['APPEARANCES'].fillna(np.nanmedian(data['APPEARANCES']))
    data['ALIVE'] = data['ALIVE'].fillna(stats.mode(data['ALIVE'])[0][0])

    
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
    data['ALIGN'] = data['ALIGN'].map({'Good Characters': 0, 'Bad Characters':1})
    return data


def new_cleaning(data):
    np.random.seed(1)
    data = data.reset_index().drop(columns=['index'])
    data['HAIR'][data['EYE'] == 'Auburn Hair'] = data['HAIR'][data['EYE'] == 'Auburn Hair'].fillna('Auburn Hair')
    data['EYE'][data['EYE'] == 'Auburn Hair'] = np.nan
    data['HAIR'][data['HAIR']=='Light Brown Hair'] = 'Brown Hair'
    data['HAIR'][data['HAIR']=='Reddish Brown Hair'] = 'Auburn Hair'
    data['HAIR'][data['HAIR']=='Reddish Blond Hair'] = 'Strawberry Blond Hair'
    data['SEX'][data['SEX']=='Genderless Characters'] = 'Agender Characters'
    data['HAIR'][data['HAIR']=='Bald'] = 'No Hair'
    data['ID'] = data['ID'].fillna('No Dual Identity')
    data['SEX'] = data['SEX'].fillna('Agender Characters')
    data['HAIR'] = data['HAIR'].fillna('No Hair')
    data['EYE'] = data['EYE'].fillna('Unknown Eyes')
    
    data['Origin'] = data['Origin'].fillna('Human')
    data['Origin'][~data['Origin'].isin(['Human','Pet','Mutant','Robot'])]='Non-Human'
    
    data['Citizenship'] = data['Citizenship'].fillna('Unknown Citizenship')
    data['Citizenship'][data['Citizenship']=='Robot']='Unknown Citizenship'
    data['Citizenship'][data['Citizenship']=='None']='Unknown Citizenship'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'American')]='American'
    data['Citizenship'][data['Citizenship']=='America']='American' 
    data['Citizenship'][data['Citizenship']=='Confederate States of America']='American'
    data['Citizenship'][data['Citizenship']=='USA']='American'
    data['Citizenship'][data['Citizenship']=='United STates']='American'
    data['Citizenship'][data['Citizenship']=='Untied States']='American'
    data['Citizenship'][data['Citizenship']=='England']='British'
    data['Citizenship'][data['Citizenship']=='Lyonesse']='British'
    data['Citizenship'][data['Citizenship']=='Strontian']='British'
    data['Citizenship'][data['Citizenship']=='Englsih']='British'
    data['Citizenship'][data['Citizenship']=='Tír na nÓg']='Irish'
    data['Citizenship'][data['Citizenship']=='China']='Chinese'
    data['Citizenship'][data['Citizenship']=='Tibetan']='Chinese'
    data['Citizenship'][data['Citizenship']=='Xian']='Chinese'
    data['Citizenship'][data['Citizenship']=='Cuba']='Cuban'
    data['Citizenship'][data['Citizenship']=='Austria']='Austrian'
    data['Citizenship'][data['Citizenship']=='Switzerland']='Swiss'
    data['Citizenship'][data['Citizenship']=='Brazil']='Brazilian'
    data['Citizenship'][data['Citizenship']=='Ireland']='Irish'
    data['Citizenship'][data['Citizenship']=='Canada']='Canadian'
    data['Citizenship'][data['Citizenship']=='Britain']='British'
    data['Citizenship'][data['Citizenship']=='Mexico']='Mexican'
    data['Citizenship'][data['Citizenship']=='South Africa']='South African'
    data['Citizenship'][data['Citizenship']=='Zulu']='South African'
    data['Citizenship'][data['Citizenship']=='Trojan']='Greek'
    data['Citizenship'][data['Citizenship']=='Carpasian']='Greek'
    data['Citizenship'][data['Citizenship']=='Lycia']='Greek'
    data['Citizenship'][data['Citizenship']=='Macedonian']='Greek'
    data['Citizenship'][data['Citizenship']=='Thracian']='Greek'
    data['Citizenship'][data['Citizenship']=='Poland']='Polish'
    data['Citizenship'][data['Citizenship']=='Norwegian']='Scandinavian'
    data['Citizenship'][data['Citizenship']=='Swedish']='Scandinavian'
    data['Citizenship'][data['Citizenship']=='Danish']='Scandinavian'
    data['Citizenship'][data['Citizenship']=='Icelandic']='Scandinavian'
    data['Citizenship'][data['Citizenship']=='Vikings']='Scandinavian'
    data['Citizenship'][data['Citizenship']=='Latveria']='Latverian'
    data['Citizenship'][data['Citizenship']=='Bavaria']='German'
    data['Citizenship'][data['Citizenship']=='Lichtenbader']='German'
    data['Citizenship'][data['Citizenship']=='Iran']='Persian'
    data['Citizenship'][data['Citizenship']=='Iranian']='Persian'
    data['Citizenship'][data['Citizenship']=='Aztec']='Native American'
    data['Citizenship'][data['Citizenship']=='Aztek']='Native American'
    data['Citizenship'][data['Citizenship']=='Mayan']='Native American'
    data['Citizenship'][data['Citizenship']=='Incan']='Native American'
    data['Citizenship'][data['Citizenship']=='Anasazi']='Native American'
    data['Citizenship'][data['Citizenship']=='Apache']='Native American'
    data['Citizenship'][data['Citizenship']=='Comanche']='Native American'
    data['Citizenship'][data['Citizenship']=='Comanches']='Native American'
    data['Citizenship'][data['Citizenship']=='Crow Nation']='Native American'
    data['Citizenship'][data['Citizenship']=='Cheyenne']='Native American'
    data['Citizenship'][data['Citizenship']=='Blackfoot']='Native American'
    data['Citizenship'][data['Citizenship']=='Elk Tribe']='Native American'
    data['Citizenship'][data['Citizenship']=='Inuit']='Native American'
    data['Citizenship'][data['Citizenship']=='Kiowa tribe']='Native American'
    data['Citizenship'][data['Citizenship']=='Utabi Tribe']='Native American'
    data['Citizenship'][data['Citizenship']=='Unnamed Tribe']='Native American'
    data['Citizenship'][data['Citizenship']=='Bonia and Herzegovina']='Bosnian'
    data['Citizenship'][data['Citizenship']=='Jamaica']='Jamaican'    
    data['Citizenship'][data['Citizenship']=='Aerie']='Aerian'
    data['Citizenship'][data['Citizenship']=='Afghan']='Afghani'
    data['Citizenship'][data['Citizenship']=='Algerians']='Algerian'
    data['Citizenship'][data['Citizenship']=='Aquilonia']='Aquilonian'
    data['Citizenship'][data['Citizenship']=='Assyrian']='Syrian'
    data['Citizenship'][data['Citizenship']=='Azanians']='Azanian'
    data['Citizenship'][data['Citizenship']=='Babylonian']='Iraqi'
    data['Citizenship'][data['Citizenship']=='Sumerian']='Iraqi'
    data['Citizenship'][data['Citizenship']=='Baluur']='Baluurian'
    data['Citizenship'][data['Citizenship']=='Belgium']='Belgian'
    data['Citizenship'][data['Citizenship']=='Canaanite']='Israeli'
    data['Citizenship'][data['Citizenship']=='Czechoslovakian']='Czech'
    data['Citizenship'][data['Citizenship']=='Frank']='French'
    data['Citizenship'][data['Citizenship']=='Kushite']='Egyptian'
    data['Citizenship'][data['Citizenship']=='Lemuran']='Lemurian'
    data['Citizenship'][data['Citizenship']=='Luminian']='Lumina'
    data['Citizenship'][data['Citizenship']=='Mbangawi']='Mbangawian'
    data['Citizenship'][data['Citizenship']=='Turkey']='Turkish'
    data['Citizenship'][data['Citizenship']=='Ottoman Empire']='Turkish'
    data['Citizenship'][data['Citizenship']=='Polemachian']='Polemachus'
    data['Citizenship'][data['Citizenship']=='San Diablo']='San Diabloan'
    data['Citizenship'][data['Citizenship']=='Saurians']='Saurian'
    data['Citizenship'][data['Citizenship']=='Tanzania']='Tanzanian'
    data['Citizenship'][data['Citizenship']=='Uganda']='Ugandan'
    data['Citizenship'][data['Citizenship']=='Ukraine']='Ukranian'
    data['Citizenship'][data['Citizenship']=='Hades']='Hell'
    data['Citizenship'][data['Citizenship']=='Inida']='India'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'United States')]='American'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'U.S.A')]='American'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Canadian')]='Canadian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'English')]='British'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Scottish')]='British'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Welsh')]='British'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'United Kingdom')]='British'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'British')]='British'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'French')]='French'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'France')]='French'   
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Japan')]='Japanese'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Asgard')]='Asgardian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Soviet')]='Russian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Russia')]='Russian'    
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Atlantis')]='Atlantean'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Atlantean')]='Atlantean'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Israel')]='Israeli'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Spain')]='Spanish'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Skrull')]='Skrull Empire'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Vietnam')]='Vietnamese'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Australia')]='Australian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Egypt')]='Egyptian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Greece')]='Greek'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Roman')]='Italian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'German')]='German'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Italy')]='Italian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Florentine')]='Italian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Florence')]='Italian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'India')]='Indian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Iraq')]='Iraqi'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Hungary')]='Hungarian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Olympia')]='Olympus'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Xandar')]='Xandarian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Haiti')]='Haitian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Hong Kong')]='Chinese'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Korea')]='Korean'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Argentina')]='Argentinian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Congo')]='Congolese'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Genosha')]='Genoshan'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Greek')]='Greek'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Sakaar')]='Sakaaran'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Savage')]='Savage Lander'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Saudi Arab')]='Saudi Arabian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Arabia')]='Saudi Arabian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Irish')]='Irish'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Celestial')]='Celestial'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Life-Model')]='Life-Model Decoy'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Madripoor')]='Madripoorian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'New Zealand')]='New Zealander'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Sparta')]='Greek'
    data['Citizenship'][data['Citizenship'].str.contains(pat = "K'un")]="K'un-Lun"
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Kree')]='Kree Empire'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Brood')]='Brood'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Deviant Lemuria')]='Deviant Lemuria'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Mesopotamian')]='Iraqi'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Lemuria')]='Lemurian'
    data['Citizenship'][data['Citizenship'].str.contains(pat = 'Wakanda')]='Wakandan'
    
    data['Marital Status'] = data['Marital Status'].fillna('Unknown Relationship')
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'Single')]='Single'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'Widowed')]='Widowed'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'widowed')]='Widowed'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'Divorced')]='Divorced'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'divorced')]='Divorced'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'Separated')]='Separated'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'Married')]='Married'
    data['Marital Status'][data['Marital Status'].str.contains(pat = 'single')]='Single'

    data = data.rename(columns={'GSM':'LGBT'})
    data['LGBT'] = data['LGBT'].map({np.nan:0})
    data['LGBT'] = data['LGBT'].map({np.nan:1, 0:0})
    #data['ALIGN'][data['ALIGN']=='Reformed Criminals'] = 'Neutral Characters'
    data = data[data['ALIGN'] != 'Neutral Characters']
    data = data[data['ALIGN'] != 'Reformed Criminals']
    data = data[pd.notnull(data['ALIGN'])]
    data = data[data['name'] != 'GenderTest']
    data['ALIVE'] = data['ALIVE'].map({'Deceased Characters':0, 'Living Characters':1})
    data.set_index('name', inplace = True)
    data = data.drop(columns=['urlslug','FIRST APPEARANCE'])
    
    data['YEAR'] = data['YEAR'].fillna(np.nanmedian(data['YEAR']))
    data['APPEARANCES'] = data['APPEARANCES'].fillna(np.nanmedian(data['APPEARANCES']))
    data['ALIVE'] = data['ALIVE'].fillna(stats.mode(data['ALIVE'])[0][0])
    
    data['Citizenship'][data['Citizenship'].isin(data['Citizenship'].value_counts()[data['Citizenship'].value_counts()<5].keys())]='Unknown Citizenship'

    
    data = data.dropna()
    data = data.join(pd.get_dummies(data['ID']))
    data = data.drop(columns=['ID'])
    data = data.join(pd.get_dummies(data['EYE']))
    data = data.drop(columns=['EYE'])
    data = data.join(pd.get_dummies(data['HAIR']))
    data = data.drop(columns=['HAIR'])
    data = data.join(pd.get_dummies(data['SEX']))
    data = data.drop(columns=['SEX'])
    data = data.join(pd.get_dummies(data['Origin']))
    data = data.drop(columns=['Origin'])
    data = data.join(pd.get_dummies(data['Citizenship']))
    data = data.drop(columns=['Citizenship'])
    data = data.join(pd.get_dummies(data['Marital Status']))
    data = data.drop(columns=['Marital Status'])
    #data = data.drop(columns=['Identity Unknown','Black Eyeballs','Platinum Blond Hair','Genderfluid Characters'])
    data['ALIGN'] = data['ALIGN'].map({'Good Characters': 0, 'Bad Characters':1})
    return data


"""

def cleaning_w_impute(data):
    np.random.seed(1)
    data = data.reset_index().drop(columns=['index'])
    data['HAIR'][data['EYE'] == 'Auburn Hair'] = data['HAIR'][data['EYE'] == 'Auburn Hair'].fillna('Auburn Hair')
    data['EYE'][data['EYE'] == 'Auburn Hair'] = np.nan
    data['HAIR'][data['HAIR']=='Light Brown Hair'] = 'Brown Hair'
    data['HAIR'][data['HAIR']=='Bald'] = 'No Hair'
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
    
    data['YEAR'] = data['YEAR'].fillna(np.nanmedian(data['YEAR']))
    data['APPEARANCES'] = data['APPEARANCES'].fillna(np.nanmedian(data['APPEARANCES']))
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
    
"""