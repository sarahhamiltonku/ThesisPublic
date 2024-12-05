

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2
import glob
import shutil


list = [
'248192-IMAGE_RGB_083_224x224_664dd0_359dd0_37dd83324619999999_-2dd2916897500000037_133dd0_224x224.png',
'232512-IMAGE_RGB_083_224x224_632dd0_336dd0_35dd1665806_-0dd3750238500000016_133dd0_224x224.png',
'249048-IMAGE_RGB_083_224x224_666dd0_348dd0_37dd99991279999999_-1dd3750234500000005_133dd0_224x224.png',
'257455-IMAGE_RGB_083_224x224_687dd0_344dd0_39dd7499121_-1dd041690250000002_133dd0_224x224.png',
'260205-IMAGE_RGB_083_224x224_696dd0_330dd0_40dd49991179999999_0dd1249759499999996_133dd0_224x224.png',
'257146-IMAGE_RGB_083_224x224_686dd0_372dd0_39dd6665788_-3dd375022650000002_133dd0_224x224.png',
'235193-IMAGE_RGB_083_224x224_637dd0_294dd0_35dd58324709999999_3dd12497475_133dd0_224x224.png',
'241729-IMAGE_RGB_083_224x224_650dd0_320dd0_36dd66658_0dd9583089499999992_133dd0_224x224.png'
]



'0-IMAGE_RGB_083_224x224_0dd0_154dd0_-17dd500065000000006_14dd79163675_217dd0_224x224.png'
'102578-IMAGE_RGB_083_224x224_404dd0_236dd0_16dd166588199999996_7dd9583061499999985_50dd0_224x224.png'

source_folder = '../../../mnt/common_drive/Satellites/'
#source_folder = '../../../mnt/common_drive/281k_10x10/281k'



source_files = os.listdir(source_folder)
print("Contents of source folder:")
print(source_files[:10])  # Print the first 10 files for brevity






print(os.path.getsize(source_folder))

output_folder = 'USE/SAT'
for image_name in list:
    source_path = os.path.join(source_folder, image_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, output_folder)
        print(f"Copied: {image_name}")
    else:
        print(f"Image not found: {image_name}")

print("Image copying completed.")


print(os.path.getsize(source_folder))

