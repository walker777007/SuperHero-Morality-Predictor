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
list_url = 'https://marvel.fandom.com/api/v1/Articles/List?category=characters&limit=100000'
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
        characters.append("https://marvel.fandom.com"+item['url'])
        page_ids.append(item['id'])
        
    offset = '&offset=' + response['offset']
        
except Exception as exc:
  print(str(exc))
  pass

#characters.pop(4644)
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
            
        if 'Earth-616 Characters' in categories:
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
            if 'Humans (Homo sapiens)' in categories:
                origins.append('Human')
            elif 'Robots' in categories:
                origins.append('Robot')        
            elif 'Mutants (Homo superior)' in categories:
                origins.append('Mutant')
            elif 'Horses' in categories:
                origins.append('Pet')     
            elif 'Pets' in categories:
                origins.append('Pet')             
            elif 'Origin' in attributes:
                if ('Human' in div[np.where(attributes=='Origin')[0][0]].div.text):
                    origins.append('Human')
                elif div[np.where(attributes=='Origin')[0][0]].div.text.strip()=='Mutant':
                    origins.append('Mutant')
                elif ('Robot' in div[np.where(attributes=='Origin')[0][0]].div.text): 
                    origins.append('Robot')
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
marvel = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/marvel-wikia-data-edited.csv')
marvel = marvel.rename(columns={'Year': 'YEAR'})
df = pd.DataFrame(list(zip(new_page_ids,names,maritals,citizenships,origins)), 
               columns =['page ids','names','Marital Status','Citizenship','Origin']) 
#%%
df = df.set_index('names')
marvel = marvel.set_index('name')
new_marvel = marvel.join(df)
#%%
new_marvel.to_csv('marvel.csv')