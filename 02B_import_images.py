"""
Script to import images from folder Data/Images/Kenya_raw and save them in a numpy array

"""

#%%
# Import  libraries
import numpy as np
import pandas as pd
import os
import cv2
import glob
import tqdm
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


#%%
# Get file path
#path = 'Data/Images/Kenya2019_raw/*.tif'
path = '../../../mnt/ext_drive/Satellites/*.png'

# Make dict File number key and File path as value
file_paths = glob.glob(path)
file_dict = {file.replace('../../../mnt/ext_drive/Satellites/', ''): file for file in file_paths}
#file_dict = {filename.split('-IMAGE')[0]: file for filename, file in file_dict.items()}




# Data: imagefile_names.csv

#Imagenames_df = pd.read_csv('Data/Imagefile_names.csv')
#Imagenames_df = pd.read_csv('Data/Kenya_Features.csv')

Imagenames_df = pd.read_csv('/../../../mnt/ext_drive/full_features_10km_27_05.csv')
print(Imagenames_df.columns)
print(Imagenames_df.shape)



print(' ---- ')
print(len(Imagenames_df))
print(len(file_dict))


print( ' ---- ')
print(' Checking files in dictionary and filenames merge')
filenames_set = set(Imagenames_df['filenames'])
file_dict_keys = set(file_dict.keys())

missing_in_file_dict = filenames_set - file_dict_keys  
missing_in_filenames = file_dict_keys - filenames_set  

print(f"Count of filenames missing in file_dict: {len(missing_in_file_dict)}")
print("Filenames missing in file_dict:", missing_in_file_dict)

print(f"\nCount of file_dict keys missing in filenames: {len(missing_in_filenames)}")
print("file_dict keys missing in filenames:", missing_in_filenames)


# Make dictionary


full_file_dict = {filename: file for filename, file in file_dict.items() if filename in filenames_set}

print(len(full_file_dict))
print('---')

random_key = (random.choice(list(full_file_dict.keys())))
print(random_key)
print(full_file_dict[random_key])


# Filter dict to just kenya

pattern = re.compile(r"133dd0_224x224.png$")


kenya_file_dict = {k: v for k, v in full_file_dict.items() if pattern.search(k)}

print('LENGTH kenya file dict')
print(len(kenya_file_dict))
random_key = (random.choice(list(kenya_file_dict.keys())))
print(random_key)
print(kenya_file_dict[random_key])


print('Length kenya image codes df')
Imagenames_df_ken = Imagenames_df[Imagenames_df['filenames'].str.contains(pattern, regex=True)]
print(len(Imagenames_df_ken))

print('----')
print('----')
print('----')

#%%

def load_image(filename, file):
    return filename, cv2.imread(file, 1)
# %%
# Read in each images and store in a dictionary
image_dic = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(load_image, filename, file): filename for filename, file in kenya_file_dict.items()}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Loading images"):
        filename, pic = future.result()
        image_dic[filename] = pic

# %%
# Print Dimensions
print('Number of images:', len(image_dic))
print('Image shape:', image_dic[random_key].shape)    


Imagenames_df_ken = Imagenames_df[Imagenames_df['filenames'].str.contains(pattern, regex=True)]
print(len(Imagenames_df_ken))


# %%
# Save Image Dictionary
np.save('Data/Arrays/Full_image_array.npy', image_dic)


# Check all images in file

pd.reset_option('display.max_colwidth')

print('\n\n\nCheck all images in numpy array:')
print(Imagenames_df_ken['filenames'].iloc[5])


random_key = (random.choice(list(image_dic.keys())))
print(random_key)
print(image_dic[random_key])


# Check all keys are present
# Convert dictionary keys and filenames column to sets
image_dic_keys = set(image_dic.keys())
filenames_set = set(Imagenames_df_ken['filenames'])

# Check if any filenames are missing in image_dic
missing_in_image_dic = filenames_set - image_dic_keys
missing_in_filenames = image_dic_keys - filenames_set 

if missing_in_filenames:
    print("These filenames in image_dic are missing from Image_names_df_ken['filenames']:")
    print(missing_in_filenames)
else:
    print("All filenames in image_dic are present in Image_names_df_ken['filenames'].")

if missing_in_image_dic:
    print("\nThese filenames in Image_names_df_ken['filenames'] are missing from image_dic keys:")
    print(missing_in_image_dic)
else:
    print("\nAll filenames in Image_names_df_ken['filenames'] are present in image_dic keys.")


print('Script run array in Data Arrays folder (Full_image_array.npy)')


