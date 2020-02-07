# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:19:29 2020

@author: walke
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:16:51 2020

@author: walke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os.path, sys, csv, requests, re
from bs4 import BeautifulSoup
import urllib
#%%
characters=[]
page_ids = []

offset = ''
list_url = 'https://dc.fandom.com/api/v1/Articles/List?category=characters&limit=100000'
#Creates data structure
char_dict = defaultdict(set)
universes = set()
  
#Repeats until HTTP error (out of pages to scrape)
try:
  while True:
    #Requests data starting from the desired offset
    r = requests.get(list_url + offset)
    response = r.json()
    
    #For each article in the response JSON:
    for item in response['items']:
        characters.append("https://dc.fandom.com"+item['url'])
        page_ids.append(item['id'])
        
    offset = '&offset=' + response['offset']
        
except Exception as exc:
  print(str(exc))
  pass
#%%
marvel_df = pd.DataFrame({'page_id':[],'name':[],'date':[]})
maritals=[]
citizenships=[]
origins=[]
names=[]
new_page_ids=[]
i=1
j=0
for char in range(len(characters)):
    categories=[]
    attributes=[]
    current_url = characters[char]
    try:
        soup=BeautifulSoup(urllib.request.urlopen(current_url).read()) 
        
        li = soup.find_all("li", {"class": "category normal"})
        for line in li:
            categories.append(line.text.strip())
            
        if 'New Earth Characters' in categories:
            names.append(soup.h1.text.strip())
            new_page_ids.append(page_ids[j])
            
            div = soup.find_all("div", {"class": "pi-item pi-data pi-item-spacing pi-border-color"})
            for line in div:
                attributes.append(line.h3.text.strip())
            attributes = np.array(attributes)
            if 'Marital Status' in attributes:
                maritals.append(div[np.where(attributes=='Marital Status')[0][0]].div.text.strip())
            else:
                maritals.append(np.nan)
            if 'Citizenship' in attributes:
                citizenships.append(div[np.where(attributes=='Citizenship')[0][0]].div.text.split(',')[0].strip())
            else:
                citizenships.append(np.nan)
            if 'Robots' in categories:
                origins.append('Robot')
            elif 'Androids' in categories:
                origins.append('Robot')
            elif 'Aliens' in categories:
                origins.append('Non-Human')
            elif 'Horses' in categories:
                origins.append('Pet')
            elif 'Pets' in categories:
                origins.append('Pet')
            elif 'Race' in attributes:
                    origins.append(div[np.where(attributes=='Race')[0][0]].div.text.strip())
            elif 'Apokoliptians' in categories:
                origins.append('Non-Human')
            elif 'Atlanteans' in categories:
                origins.append('Non-Human')
            elif 'Amazons' in categories:
                origins.append('Non-Human')
            elif 'Peak Human Condition' in categories:
                origins.append('Human')
            elif 'Latinos' in categories:
                origins.append('Human')
            elif 'Vietnamese' in categories:
                origins.append('Human')
            elif 'Chinese' in categories:
                origins.append('Human')
            elif 'Japanese' in categories:
                origins.append('Human')
            elif 'Russians' in categories:
                origins.append('Human')
            elif 'Brits' in categories:
                origins.append('Human')
            elif 'Germans' in categories:
                origins.append('Human')
            elif 'French' in categories:
                origins.append('Human')
            elif 'Exposure to Chemicals or Radiation' in categories:
                origins.append('Human')
            elif 'Americans' in categories:
                origins.append('Human')
            elif 'Blue Skin' in categories:
                origins.append('Non-Human')
            elif 'Grey Skin' in categories:
                origins.append('Non-Human')
            elif 'Red Skin' in categories:
                origins.append('Non-Human')
            elif 'Green Skin' in categories:
                origins.append('Non-Human')
            elif 'Purple Skin' in categories:
                origins.append('Non-Human')
            elif 'Origin' in attributes:
                if ('Human' in div[np.where(attributes=='Origin')[0][0]].div.text):
                    origins.append('Human')
                else:
                    origins.append('Non-Human') 
            else:
                origins.append(np.nan)
        
            print(j+1,i,current_url)
            i+=1
        j+=1
        
    except:
        j+=1
#%%
dc = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/dc-wikia-data-edited.csv')
df = pd.DataFrame(list(zip(new_page_ids,names,maritals,citizenships,origins)), 
               columns =['page ids','names','Marital Status','Citizenship','Origin']) 
#%%
df = df.set_index('page ids')
dc = dc.set_index('page_id')
df = df.set_index('names')
dc = dc.set_index('name')
new_dc = dc.join(df)
#%%
new_dc.to_csv('dc.csv')