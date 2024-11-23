"""
Preprocessing survey data and get averages of clusters 

TODO: Look at Yeh. Chris 
TODO: More work here ot clean find best variables etc. 

"""


#%%
# Libraries

import pandas as pd
import numpy as np
import os


#%%


#%% 
# Load Data
Clustered_df = pd.read_csv('Data/clustered_full.txt', sep='\t')
# %%

# %%

print(Clustered_df.value_counts('avg_roofcat'))

print(Clustered_df.value_counts('avg_wealthscore'))
# %%


# Save Clustered Df with just what I want for now


Clustered_df = Clustered_df[['latitude', 'longitude', 'household_count', 
                            'avg_roofcat', 'avg_wealthscore' ,'year']]

Clustered_df.to_csv('Data/clean_clustered_full.txt', sep='\t', index=False)
# %%

print('03B done successfully :)')