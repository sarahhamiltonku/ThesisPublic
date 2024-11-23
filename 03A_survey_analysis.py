"""
DHS survey analysis 

So far serves 2 purposes
1. Basic analysis of DHS data
2. Prep in clusters for connecting with satellite data numbers (can also do this 03A)

Uses describ_env enviornment

"""


#%%
# Import libraries
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from plotnine import *
import sys

#%%

# Load data

df_full = pd.read_csv('Data/Household.csv')
df = df_full[df_full['countryname']=='Kenya']

print('DHS data length: ', str(len(df)))

# %%

# Remove outputs without latitude or longitude 


df = df.dropna(subset=['latitude'])
df = df.dropna(subset=['latitude'])

print('After empty lat and long: ', str(len(df)))
print(df.value_counts('year'))


print(df.value_counts('surveytype'))
# %%

# Save heatmaps of the datapoints I have for each year

for year in df['year'].unique():
  print(year)
  df_year = df[df['year']==year]
  print(len(df_year))

  map_center = [df_year['latitude'].mean(), df_year['longitude'].mean()]
  m = folium.Map(location=map_center, zoom_start=5)
  heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
  HeatMap(heat_data).add_to(m)

  output_file = 'Data/Descriptives/heatmaps/survey_data'+str(year)+'.html'
  m.save(output_file)



#%% 

# Visualise data

pal = ['#7bc043', '#0392cf']
pal2 = ['#7bc043', '#59a9a4', '#4286f4', '#4169a3', '#0392cf']
df['year_cat'] = df['year'].astype('str')
year_order = ['2003', '2008', '2014', '2015']
# %%

df_dropped = df.dropna(subset=['roofcat'])
# 29 missing roofcats


# Roof category

labels = ['Non-adequate', 'Adequate']

plot = (
    ggplot(df_dropped, aes(x='factor(year_cat)', fill='factor(roofcat)'))  
    + geom_bar(position = 'fill') 
    + scale_x_discrete(limits=year_order) 
    + scale_fill_manual(values=pal, labels=labels)
    + labs(title='Roof Categories by Year', x='Year', y='Proportion', fill='Roof Category')
    + theme_minimal()
)

print(plot)
# %%

# wall category

df_dropped = df.dropna(subset=['wallcat'])

labels = ['0', '1']

plot = (
    ggplot(df_dropped, aes(x='factor(year_cat)', fill='factor(wallcat)'))  
    + geom_bar(position = 'fill') 
    + scale_x_discrete(limits=year_order) 
    + scale_fill_manual(values=pal, labels=labels)
    + labs(title='Wall Categories by Year', x='Year', y='Proportion', fill='Wall Category')
    + theme_minimal()
)

print(plot)
# 8578 missing roofcats
# %%
# Urban Rural 


df_dropped = df.dropna(subset=['urbanrural'])

labels = ['Rural', 'Urban']

plot = (
    ggplot(df_dropped, aes(x='factor(year_cat)', fill='factor(urbanrural)'))  
    + geom_bar(position = 'fill') 
    + scale_x_discrete(limits=year_order) 
    + scale_fill_manual(values=pal, labels=labels)
    + labs(title='Urban Rural by Year', x='Year', y='Proportion', fill='')
    + theme_minimal()
)

print(plot)

# %%

# Agriculture Land


df_dropped = df.dropna(subset=['agland'])
# 8571 missing 
labels = ['Non-agricultural', 'Agricultural']

plot = (
    ggplot(df_dropped, aes(x='factor(year_cat)', fill='factor(agland)'))  
    + geom_bar(position = 'fill') 
    + scale_x_discrete(limits=year_order) 
    + scale_fill_manual(values=pal, labels=labels)
    + labs(title='Agricultural Land by Year', x='Year', y='Proportion', fill='')
    + theme_minimal()
)

print(plot)
# %%

# Wealth index 

df_dropped = df.dropna(subset=['wealthindex'])
# 8571 missing 
#labels = ['Non-agricultural', 'Agricultural']

plot = (
    ggplot(df_dropped, aes(x='factor(year_cat)', fill='factor(wealthindex)'))  
    + geom_bar(position = 'fill') 
    + scale_x_discrete(limits=year_order) 
    + scale_fill_manual(values=pal2)
    + labs(title='Wealth Index by Year', x='Year', y='Proportion', fill='')
    + theme_minimal()
)

print(plot)



# %%

# Clusters 


df_grouped = df.groupby(['year', 'clusterid']).size().reset_index(name='household_count')


plot_box = (
    ggplot(df_grouped, aes(x='factor(year)', y='household_count'))  
    + geom_boxplot()
    + labs(title='Cluster (village) number of surveyed households', 
           x='Year', y='Household Count')
    + theme_minimal()
)

# Print the plot
print(plot_box)

# %%


df_clusters = df.groupby(['year', 'clusterid']).agg(
    mean_latitude=('latitude', 'mean'),
    mean_longitude=('longitude', 'mean'),
    latitude_all=('latitude', lambda x: list(x)),
    longitude_all=('longitude', lambda x: list(x)),
    household_count=('clusterid', 'count')
).reset_index()


# All lat and long in each cluster is the same 
# %%
# Check no missing data from dropped lats and longs
"""
df_full = pd.read_csv('data/Household.csv')
df_ = df_full[df_full['countryname']=='Kenya']


df_clusters2 = df.groupby(['year', 'clusterid']).agg(
    mean_latitude=('latitude', 'mean'),
    mean_longitude=('longitude', 'mean'),
    latitude_all=('latitude', lambda x: list(x)),
    longitude_all=('longitude', lambda x: list(x)),
    household_count=('clusterid', 'sum')
).reset_index()


df_clusters2.dropna(subset=['mean_latitude'], inplace=True)
"""
# %%

# Remake without list 

df_clusters = df_clusters.drop(columns=['latitude_all', 'longitude_all'])
df_clusters = df_clusters.rename(columns={'mean_latitude': 'latitude', 'mean_longitude': 'longitude'})
df_clusters = df_clusters[(df_clusters['latitude'] != 0.0) & (df_clusters['longitude'] != 0.0)]




# %%
# Remake maps with cluster data 

import matplotlib.cm as cm
import matplotlib.colors as colors


norm = colors.Normalize(vmin=df_clusters['household_count'].min(), vmax=df_clusters['household_count'].max())
colormap = cm.ScalarMappable(norm=norm, cmap='YlGnBu')  

def get_color(household_count):
    rgba_color = colormap.to_rgba(household_count)
    return colors.to_hex(rgba_color)

for year in df_clusters['year'].unique():
    print(year)
    df_year = df_clusters[df_clusters['year']==year]
    print(len(df_year))

    map_center = [df_year['latitude'].mean(), df_year['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=5)

    for _, row in df_year.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius= 5 ,
            color=get_color(row['household_count']),  # Color based on household_count
            fill=True,
            fill_color=get_color(row['household_count']),
            fill_opacity=0.5,
            popup=f"Cluster: {row['clusterid']}, Households: {row['household_count']}"
        ).add_to(m)
    
    output_file = f'Data/Descriptives/heatmaps/survey_data_clustered_{year}.html'
    m.save(output_file)
# %%

df_clusters.to_csv('Data/clustered_latlongs.txt', sep='\t', index=False)

# Looking at Clusters to Satelite Images 


# %%

# Save DF clusters 


df_clusters_full = df.groupby(['year', 'clusterid']).agg(
    mean_latitude=('latitude', 'mean'),
    mean_longitude=('longitude', 'mean'),
    latitude_all=('latitude', lambda x: list(x)),
    longitude_all=('longitude', lambda x: list(x)),
    household_count=('clusterid', 'count'),
    avg_roofcat=('roofcat', 'median'),
    avg_wealthscore=('wealthindex', 'median'),
    roofcat=('roofcat', lambda x: list(x)),
    wealthindex=('wealthindex', lambda x: list(x)),
    wealthscore=('wealthscore', lambda x: list(x)),
    hhweight=('hhweight', lambda x: list(x))
).reset_index()





df_clusters_full = df_clusters_full.drop(columns=['latitude_all', 'longitude_all'])
df_clusters_full = df_clusters_full.rename(columns={'mean_latitude': 'latitude', 'mean_longitude': 'longitude'})
df_clusters_full = df_clusters_full[(df_clusters_full['latitude'] != 0.0) & (df_clusters_full['longitude'] != 0.0)]

df_clusters_full.to_csv('Data/clustered_full.txt', sep='\t', index=False)


# %%

script_name = sys.argv[0]  
print(f"script {script_name} complete")








