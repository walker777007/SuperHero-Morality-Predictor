# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:11:13 2020

@author: walke
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from cleaning import cleaning
import warnings
warnings.filterwarnings('ignore')
#%%
dc = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/dc-wikia-data-edited.csv')
marvel = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/SuperHero-Morality-Predictor/data/marvel-wikia-data-edited.csv')
marvel = marvel.rename(columns={'Year': 'YEAR'})

data = pd.concat([dc,marvel])
data = cleaning(data)

data.set_index('ALIGN', inplace = True)

data = data[:500]
#%%
def make_dendrogram(dataframe, linkage_method, metric, color_threshold=None):
    '''
    This function creates and plots the dendrogram created by hierarchical clustering.
    
    INPUTS: Pandas Dataframe, string, string, int
    
    OUTPUTS: None
    '''
    distxy = squareform(pdist(dataframe.values, metric=metric))
    Z = linkage(distxy, linkage_method)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
        labels = dataframe.index,
        color_threshold = color_threshold
    )
    plt.show()
#%%
linktype = 'complete'
metric = 'cityblock'
make_dendrogram(data, linktype, metric, color_threshold=None)