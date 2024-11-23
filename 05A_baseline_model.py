
# Import libraries
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import layers as cv_layers
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle
import glob
import cv2
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
print("All imports loaded successfully!")
# Check versions

print(tf.__version__)
print(keras.__version__)
print(keras_cv.__version__)
print('All run ready for datas')

# Images
image_file = os.path.join('Data', 'Arrays', 'Kenya2019_array_augmented.npy')
image_file = os.path.join('Data', 'Arrays', 'Kenya_array_augmented.npy')
images = np.load(image_file, allow_pickle=True)
images = images.item()
print(len(images))


# Labels
df = pd.read_csv(os.path.join('Data', 'clustered_geopoints.txt'), sep = '\t')
df['avg_roofcat'] = df['avg_roofcat'].replace(0.5, 0)
labels = df['avg_roofcat'].values
image_codes = df['filenames'].astype(str).values #imagecodes 
#Remove nan
non_nan = ~np.isnan(labels)
labels = labels[non_nan]
image_codes = image_codes[non_nan]


print(df['avg_roofcat'].value_counts())



X = np.array([images[code] for code in image_codes if code in images])
print(f"Image data shape: {X.shape}")
print(f"Labels shape: {labels.shape}")


X_train, X_test, y_train, y_test, image_codes_train, image_codes_test = train_test_split(X, labels, image_codes, test_size=0.2, random_state=42)


# Flat arrays
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)



knn_type = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn_type, param_grid, cv=5)

# Train
knn_gscv.fit(X_train_flat, y_train)

# Predict

y_pred = knn_gscv.predict(X_test_flat)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Baseline KNN  Mean Squared Error: {mse}")




print('Find best metrics: ')
# Grid search: parameters from kaggle reference
grid_params = { 'n_neighbors' : np.arange(1, 26), 
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}


# Add paramaeters to search
knn_grid = GridSearchCV(knn_type, grid_params, verbose = 1, cv=5, n_jobs = -1)
knn_grid.fit(X_train_flat, y_train)

print(knn_grid.best_score_)
print(knn_grid.best_params_)

# Based off best results

best_params = knn_grid.best_params_
knn_model = KNeighborsClassifier(
    n_neighbors=best_params['n_neighbors'],
    weights=best_params['weights'],
    metric=best_params['metric'],
    algorithm='brute'
)
knn_model.fit(X_train_flat, y_train)


y_pred = knn_model.predict(X_test_flat)


mse = mean_squared_error(y_test, y_pred)
print(f"KNN Classifier Mean Squared Error: {mse}")



print('Test set accuracy: ',metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix: ')
print(cm)


print('Predicted Labels: ')

class_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
print(Counter(class_counts))






# Save predictions 

predictions_df = pd.DataFrame

results_df = pd.DataFrame({
     'Image_Code': image_codes_test,  
    'Actual': y_test,    
    'Predicted': y_pred  
})

# Save to a CSV file

results_df.to_csv("Data/Results/MODEL1/knn_predictions.txt", sep='\t', index=False)







# Add visualisations


from sklearn.model_selection import cross_val_score

# Define the range of neighbors to test
neighbors = range(1, 26)
cv_scores = []

# Cross-validate for each k
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_flat, y_train, cv=5, scoring='accuracy')
    cv_scores.append(1 - scores.mean())  # Misclassification error = 1 - accuracy

# Plotting misclassification error vs. number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(neighbors, cv_scores, marker='o')
plt.xlabel('Number of K Neighbors')
plt.ylabel('Misclassification Error')
plt.title('Misclassification Error vs. K Value for KNN')
plt.savefig("Data/Results/MODEL1/K_tracking.png")

plt.tight_layout()
plt.close() 







from sklearn.metrics import roc_curve, auc

# Only applicable for binary classification
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("Data/Results/MODEL1/ROC.png")

plt.tight_layout()
plt.close() 

