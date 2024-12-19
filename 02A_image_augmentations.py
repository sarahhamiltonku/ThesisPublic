"""
# Explore and choose best image augmentations

# These augmentations are for all models (so different from 07 random augmentations)

Guide to Augmentations:

# Uses geo_env environment 
"""


#%%

#from osgeo import gdal as GD  
import numpy as np 
import cv2
import glob
import random
import os
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import albumentations as A
import random


#%%
# Image path

folder_path = '../../../mnt/ext_drive/281k_10x10/281k'

#folder_path = 'Data/Images/kenya2019_raw/'
all_files = os.listdir(folder_path)
tif_files = [file for file in all_files if file.endswith('.png')]


random.seed(999)
random_files1 = random.sample(tif_files, 16)

random.seed(23)
random_files2 = random.sample(tif_files, 16)



#%% Make output file
output_folder = 'Data/Images'
os.makedirs(output_folder, exist_ok=True)







# %%
def augmentations(image):
    # Apply all augmentations
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 2.0))
    image = ImageOps.flip(image)  # Flip vertically
    image = ImageOps.mirror(image)  # Flip horizontally
    image = ImageOps.solarize(image, threshold=random.randint(64, 192))  # Apply solarization
    
    return image
# Set up the figure for displaying a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

# Loop through each selected image, apply augmentations, and display
for i, file in enumerate(random_files2):
    image_path = os.path.join(folder_path, file)
    image = Image.open(image_path)
    
    # Apply random augmentations
    augmented_image = augmentations(image)
    
    #save_path = os.path.join(output_folder, f'augmented_{i}_{file}')
    #augmented_image.save(save_path)


    #axes[i].imshow(augmented_image)
    #axes[i].axis('off')  # Hide the axis


#plt.tight_layout()
#plt.show()

###########################################################
###########################################################
###########################################################
# %%

# Pixel Augmentations
transform = A.Compose([
    A.CLAHE(p=1),
    A.FancyPCA(p=1, alpha=1),
    
    A.RandomBrightnessContrast(p=1, contrast_limit=(0.4,0.4), brightness_limit=(0.35,0.35)),
    A.Sharpen(p=1, alpha=(0.85, 0.85)), 
    A.HueSaturationValue(p=1, hue_shift_limit=(0,0), sat_shift_limit=(10,10))
])



transform = A.Compose([
    A.CLAHE(p=1, clip_limit=3),
    #A.FancyPCA(p=1, alpha=0.6),
    A.RandomBrightnessContrast(p=1, contrast_limit=(0.2,0.2), brightness_limit=(0, 0)),
    A.Sharpen(p=1, alpha=(0.9,0.9), lightness=(0.8,0.8)),
    A.HueSaturationValue(p=1, hue_shift_limit=(0,0), sat_shift_limit=(0,0), val_shift_limit=(0,0))
])

#%%
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i, file in enumerate(random_files1):
    image_path = os.path.join(folder_path, file)
    image = np.array(Image.open(image_path))
    #.convert('RGB'))
    
    # Apply augmentations
    augmented = transform(image=image)
    augmented_image = augmented['image']

    # Grid
    axes[i].imshow(augmented_image)
    axes[i].axis('off')  

    
# Adjust layout and show the plot
plt.tight_layout()
save_path = os.path.join(output_folder, '2A_test_augmentations_set1.png')
plt.savefig(save_path, dpi=300)
plt.close(fig)

# %%

# Save pre augmentations too


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()
for i, file in enumerate(random_files1):
    image_path = os.path.join(folder_path, file)
    image = np.array(Image.open(image_path))
    #.convert('RGB'))


    # Grid
    axes[i].imshow(image)
    axes[i].axis('off')  

    
# Adjust layout and show the plot
plt.tight_layout()
save_path = os.path.join(output_folder, '2A_test_unedited_set1.png')
plt.savefig(save_path, dpi=300)
plt.close(fig)