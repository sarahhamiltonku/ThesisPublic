"""

This has partly been done in Google Collab 
Check how many images we have 
Also has been done below slightly (old 00_combine_survey_images)

Uses geo_env environment
"""
#%%
# import libraries

import pandas as pd
import geopandas as gpd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import re 
import os


output_folder = 'Data/Images'
os.makedirs(output_folder, exist_ok=True)


# %%
# load data
clusters_df = pd.read_csv('Data/clean_clustered_full.txt', sep='\t')  
#image_files = pd.read_csv('Data/imagefile_names.csv')
image_files = pd.read_csv('Data/Kenya_Features.csv')
image_files = pd.read_csv('/../../../mnt/ext_drive/full_features_10km_27_05.csv')

image_files = image_files.loc[:, ~image_files.columns.str.startswith('feature_')]

pattern = re.compile(r"133dd0_224x224.png$")
image_files = image_files[image_files['filenames'].str.contains(pattern, regex=True)]


print(len(image_files))
print(image_files)



# Import image arrays
image_arrays = np.load('Data/Arrays/Full_image_array.npy', allow_pickle=True)
image_arrays = image_arrays.item()
# %%
# Join data with long lat simple
dhs_geopoints = gpd.GeoDataFrame(clusters_df, 
geometry=gpd.points_from_xy(clusters_df.longitude, clusters_df.latitude))

dhs_geopoints = dhs_geopoints[~dhs_geopoints['geometry'].is_empty]

image_geopoints = gpd.GeoDataFrame(image_files, geometry=gpd.points_from_xy(image_files.lon, image_files.lat))

df_merge = gpd.sjoin_nearest(
    dhs_geopoints, image_geopoints, how='left', distance_col='distance' )


df_merge.rename(columns={'geometry': 'geometry_survey'}, inplace=True)    # , 'cellname': 'image_code'
df_merge[['lat', 'latitude', 'lon', 'longitude']]

print(df_merge.head())
print(df_merge.columns)

#%%
# Save geopoints
df_merge.to_csv('Data/clustered_geopoints.txt', sep='\t', index=False)

# %%
print('Number of Survey Results:')
print(len(df_merge))

print('Number of Unique Image Codes:')
print(df_merge['filenames'].nunique()) #imagecode


print('Number of Unique Image Codes in original file:')
print(image_files['filenames'].nunique()) #imagecode


#%%

#TODO: Fix deduplications

df_dedupe = df_merge.sort_values(by=['filenames', 'distance']).drop_duplicates(subset='filenames', keep='first')
print(len(df_dedupe))

print(df_dedupe['avg_roofcat'].value_counts())

df_merge = df_dedupe


df_merge.to_csv('Data/clustered_geopoints.txt', sep='\t', index=False)

# %%
# cut dict to only those with DHS 

image_array_clean = image_arrays.copy()

for key in list(image_array_clean.keys()):  
    if (key) not in df_merge['filenames'].values: #imagecode
        image_array_clean.pop(key)
# %%


#%%

# Albumentations Script 

transform = A.Compose([
    A.CLAHE(p=1),
    A.FancyPCA(p=1, alpha=1),
    A.RandomBrightnessContrast(p=1, contrast_limit=(0.45,0.45), brightness_limit=(0.35,0.35)),
    A.Sharpen(p=1, alpha=(0.85, 0.85)), 
    A.HueSaturationValue(p=1, hue_shift_limit=(0,0), sat_shift_limit=(10,10))
])


transform = A.Compose([
    A.CLAHE(p=1, clip_limit=3),
    #A.FancyPCA(p=1, alpha=0.6),
    A.RandomBrightnessContrast(p=1, contrast_limit=(0.2,0.2), brightness_limit=(0, 0)),
    A.Sharpen(p=1, alpha=(0.9,0.9), lightness=(0.8,0.8)),
    A.HueSaturationValue(p=1, hue_shift_limit=(5,5), sat_shift_limit=(0,0), val_shift_limit=(0,0))
])

#%%

# Apply augmentations to all images in image_array_clean

augmented_images = {}
for key, image in image_array_clean.items():
    augmented = transform(image=image)
    augmented_images[key] = augmented['image']
# %%
# Save augmented images
np.save('Data/Arrays/Kenya_array_augmented.npy', augmented_images)
# %%
# Display 16 random augmented images
random_files = np.random.choice(list(augmented_images.keys()), 16)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i, key in enumerate(random_files):
    image = augmented_images[key]
    axes[i].imshow(image)
    axes[i].axis('off')
# %%
   
# Adjust layout and show the plot
plt.tight_layout()
save_path = os.path.join(output_folder, '4A_traindata_augmented.png')
plt.savefig(save_path, dpi=300)
plt.close(fig)
