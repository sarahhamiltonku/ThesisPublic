"""
Download images from Google Earth Engine to Google Drive
This script comes from KC



Download guide for Earth Engine images
A Basic init
1. Create project on Google Cloud Console
2. Download google cloud sdk to location

B To start new download to drive
1. Go to https://console.cloud.google.com/ and select the correct project. Go to Dashboard
2. Go to project settings -> Service accounts
3. Create Service Account (if needed)
4. Give name, make sure to give owner access 
5. Click on the service account name when created
6. Go to keys -> Add json key. Upload it to server where you will run the python script
7. Go to the script, add the service account address to the script, as well as json file path (x2)
8. Go to google drive folder, create new folder, and share it with this service account (editor)
9. In python script, in task, folder = 'folder_from_drive'
10 In script, change batch start 
11 Screen -S ee_date 
12 conda activate ee
13 gloud init
14 Select correct account and project
15 python scriptname.py

C Whenever it resets
1. Download all images from drive
2. Store all filenames in csv (dir /b > files.csv)
3. Replace dd by . in csv, add column header files
4. Run extract_lat_longs.py (change df location + save location)
5. Run update ee_script (change df_updated, df_ac, save location)
6. Upload csv to where python script is
7. Follow instructions from B

Uses in a terminal script and with environment Geo_env?? or another one

"""


#%%
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os
import ee
import time
#from io import StringIO

print('imports done')

# Come from Tokens
service_account = "service@earth-engine2-435313.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, 'Tokens/Client_auth/earth-engine2-435313-b6a3290a2182.json')

ee.Initialize(credentials)
print('init done')

gauth = GoogleAuth()
scopes = ['https://www.googleapis.com/auth/drive']
gauth.credentials = ServiceAccountCredentials(service_account, 'Tokens/Client_auth/earth-engine2-435313-b6a3290a2182.json')

drive = GoogleDrive(gauth)



import time
#import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


africa_all_db = pd.read_csv('Data/imagefile_names.csv')

africa_all_db[1:5]

#def downloadMapDataToDrive(bandSelection = "RGB", db = africa_all_db,
#                           batchSize = len(africa_all_db), batchStart=0, resolution = 0.04166665 ,
#                           dims = '224x224'):

print(africa_all_db.head())
def downloadMapDataToDrive(bandSelection = "RGB", db = africa_all_db,
                           batchSize = 10, batchStart=0, resolution = 0.04166665 ,
                           dims = '224x224', start_date='2003-01-01', end_date='2003-12-31'): #CHANGE dates
    
    print('starting downloadMapDataToDrive at :', batchStart, flush=True)
    #These bands are reasonable, but perhaps explore others.
    if bandSelection == "RGB":
        #bands = ['B4','B3','B2'] # CHANGE bands based on Landsat version and what am doing
        bands = ['B3','B2','B1']
        #bands = ['SR_B4','SR_B3','SR_B2']
    elif bandSelection == "IR":
        bands = ['B7','B6','B5']
    else:
        print("pick custom bands")


    # Standard is LC08 t
    #collection = ee.ImageCollection('LANDSAT/LC08/C02/T1') \
    collection = ee.ImageCollection('LANDSAT/LE07/C02/T1') \
        .filterDate(start_date, end_date) 
    raster = ee.Algorithms.Landsat.simpleComposite(collection, 50, 10).select(bands)


    for i in range(batchStart, batchSize):
        lonMin =  db['lon'].iloc[i] - resolution
        latMin =  db['lat'].iloc[i] - resolution
        lonMax =  db['lon'].iloc[i] + resolution
        latMax =  db['lat'].iloc[i] + resolution
        geometry = ee.Geometry.Rectangle([lonMin, latMin, lonMax, latMax])

        geometry = geometry['coordinates'][0]


        lon=str(db['lon'].iloc[i]).replace('.', 'dd')
        lat=str(db['lat'].iloc[i]).replace('.', 'dd')
        cell_name=str(db['cellname'].iloc[i]).replace('.','dd') # uniqid
        country=str(db['country_code'].iloc[i]).replace('.','dd') #country
        fileName = str(i)+"-IMAGE_"+bandSelection+"_"+str(resolution)[2:5]+"_"+dims+"_"+str(lon)+"_"+str(lat)+"_"+str(country)+"_"+str(cell_name)

        print(fileName, flush=True)
        task = ee.batch.Export.image.toDrive(image=raster,
                                             description="imageToDrive",
                                             folder="Kenya2003", # CHANGE folder path in Google rive
                                             fileNamePrefix=fileName,
                                             dimensions=dims,
                                             region=geometry)
            
        task.start()
        time.sleep(10)


        if (i+1) % 100 == 0:
            print('sleeping to write', flush=True)
            time.sleep(10*60)
    

downloadMapDataToDrive(bandSelection = "RGB",
                       db = africa_all_db,
                       batchSize = len(africa_all_db),
                       batchStart=1,
                       resolution = 0.04166665,
                       dims = '224x224')

print('Finished')
# %%
