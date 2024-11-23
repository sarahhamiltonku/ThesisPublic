


#%%

# Open results and save with df and images sample 


# Libraries 

# Import libraries
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import cv2


# Function

def save_predictions(predictions_path, geopoints_path, clusters_path, output_path):
    model_df = pd.read_csv(predictions_path, sep='\t')
    geopoints_df = pd.read_csv(geopoints_path, sep='\t')

    model_df = pd.merge(model_df, geopoints_df, left_on='Image_Code', right_on='filenames', how='left')

    clusters_df = pd.read_csv(clusters_path, sep='\t')
    model_df = pd.merge(
        model_df, 
        clusters_df,
        #[['year', 'clusterid', 'latitude', 'longitude']], 
        on=['latitude', 'longitude'], 
        how='left'
    )

    print(f"Rows after merging: {len(model_df)}")

    # Save the processed DataFrame
    model_df.to_csv(output_path, sep='\t', index=False)

    return model_df

model3_df =save_predictions(
    #'Results/MODEL3/Final/cot_cnn_predictions.txt',
    'Results/MODEL3/round8/cot_cnn_predictions.txt',
    'Data/clustered_geopoints.txt',
    'Data/clustered_full.txt',
    'Results/MODEL3/Final_model3_predictions.txt'
)
"""
save_predictions(
    'Results/MODEL2/trans_cnn_predictions.txt',
    'Data/clustered_geopoints.txt',
    'Data/clustered_full.txt',
    'Results/MODEL2/Final_model2_predictions.txt'
)

save_predictions(
    'Results/MODEL1/knn_predictions.txt',
    'Data/clustered_geopoints.txt',
    'Data/clustered_full.txt',
    'Results/MODEL1/Final_model1_predictions.txt'
)
"""




# 2nd STEP save images 


def find_matching_images(df, predictions_column, filenames_column, num_samples=10):

    folder_path = '../../../mnt/ext_drive/Satellites'
    all_images = os.listdir(folder_path)
    matching_images = {}
    sampled_df_combined = pd.DataFrame()


    for prediction in [0, 1]:
        filtered_df = df[df[predictions_column] == prediction]


        sampled_df = filtered_df.sample(n=num_samples, random_state=42)
        filenames = sampled_df[filenames_column].tolist()

        matching_images[prediction] = [
            os.path.join(folder_path, img) for img in all_images if img in filenames
        ]
        sampled_df_combined = pd.concat([sampled_df_combined, sampled_df], ignore_index=True)

    return matching_images, sampled_df_combined


matching_images, df_latlongs = find_matching_images(
    df=model3_df,  
    predictions_column='Predicted',  
    filenames_column='Image_Code', 
    num_samples=9
)



print(df_latlongs.columns)
df_latlongs.to_csv('Results/MODEL3/latlongresults.txt', sep='\t', index=False)
df_latlongs = df_latlongs[['Image_Code', 'Actual', 'Predicted', 'latitude', 'longitude']]

print(df_latlongs)
print(df_latlongs[['latitude', 'longitude']])





def save_images(matching_images, df, filenames_column, results_folder):
    os.makedirs(results_folder, exist_ok=True)
    for prediction in [0, 1]:
        images = matching_images[prediction]

        # Load images for grid creation
        loaded_images = []
        folder_path = '../../../mnt/ext_drive/Satellites'
        for image_filename in images:
            img_path = os.path.join(folder_path, image_filename)
            img = cv2.imread(img_path)
            if img is not None:
                loaded_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        grid_filename = os.path.join(results_folder, f"prediction_{prediction}_grid.png")
        save_grid(loaded_images, grid_filename)


def save_grid(images, output_path, grid_size=(3, 3), image_size=(224, 224)):

    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = cv2.resize(images[i], image_size)
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved grid to {output_path}")


results_folder = './Results'
save_images(
    matching_images=matching_images, 
    df=model3_df,                     
    filenames_column='Image_Code',    
    results_folder=results_folder    
)


print(type(matching_images))

