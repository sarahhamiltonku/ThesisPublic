
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
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.keras.callbacks import Callback
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.regularizers import l2




# import libraries

import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import layers as cv_layers
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import pickle
import glob
import cv2
import re
import os
from collections import Counter
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt

import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import Recall
from sklearn.utils import class_weight
from tensorflow.keras import backend as K


from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model


print("All imports loaded successfully!")
# Check versions

print(tf.__version__)
print(keras.__version__)
print(keras_cv.__version__)
print('All run ready for data :)! ')

print(' ----')
tf.keras.backend.clear_session()

# Check for GPU devices
gpus = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", gpus)

# Enable memory growth for each GPU
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memory growth enabled for GPU: {gpu}")




# Preprocess
NUM_CLASSES = 1
BATCH_SIZE = 16 #8



# Tensor board

log_dir = os.path.join("logs", "fit", pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Images
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






def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)    






# Normalise images
images = {key: np.array(img).astype('float32') / 255.0 for key, img in images.items()}
X = np.array([images[code] for code in image_codes if code in images])


X_images = np.array([images[code] for code in image_codes if code in images])
X_images = X_images.astype('float32') / 255.0  # Normalize images

print(f"Image data shape: {X.shape}")
print(f"Labels shape: {labels.shape}")

print(len(X), len(labels), len(image_codes))


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


print(len(labels))


print('labels check: ')
print("Labels (first 10):", labels[:10])
print("Unique Labels:", np.unique(labels))
print("Labels Shape:", labels.shape)





# Flat arrays
X_train_flat = X_train_images.reshape(X_train_images.shape[0], -1)
X_test_flat = X_test_images.reshape(X_test_images.shape[0], -1)


def preprocess_data(images, labels):
    labels = tf.cast(labels, tf.float32)
    return images, labels 





test_dataset = tf.data.Dataset.from_tensor_slices((X_test_images, y_test))
test_dataset = test_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



backbone = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")

for _, label in train_dataset.take(5):
    print("Label:", label.numpy(), "Shape:", label.numpy().shape)


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
#Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict[0] *= 2



kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
cv_results = []
for train_idx, val_idx in kfold.split(X_train_images, y_train):
    print(f"Training fold {fold_no}...")

    train_images = X_train_images[train_idx]
    train_labels = y_train[train_idx]

    val_images = X_train_images[val_idx]
    val_labels = y_train[val_idx]




    train_dataset = train_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    val_dataset = val_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



    activation = "sigmoid"
loss = 'binary_crossentropy'


model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES, 
    activation=activation, #sigimoide
)



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)


model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=[f1_score, Recall(class_id=0)]
    #metrics=['accuracy']

)

model.summary()

# FIT
history =  model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    verbose =1,
    class_weight=class_weights_dict,
    callbacks=[tensorboard_callback,lr_schedule, early_stopping]  
)



val_loss, val_f1_score, val_recall_class_0 = model.evaluate(val_dataset)
print(f"Validation results for fold {fold_no}: Loss={val_loss}, F1 Score={val_f1_score}, Recall for Class 0={val_recall_class_0}")
    
cv_results.append({'fold': fold_no, 'val_loss': val_loss, 'f1_score': val_f1_score, 'recall_class_0': val_recall_class_0})
fold_no += 1





print("\nFinal evaluation on the test dataset:")
test_loss, test_f1_score, test_recall_class_0 = model.evaluate(test_dataset)
print(f"Test results: Loss={test_loss}, F1 Score={test_f1_score}, Recall for Class 0={test_recall_class_0}")

print(cv_results)



##### 





y_pred = model.predict(test_dataset).flatten()
y_pred_percent = y_pred  # Raw Probs
y_pred_binary = (y_pred > 0.70).astype(int)  


# Metrics calculation
mae = mean_absolute_error(y_test, y_pred_binary)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred_binary)
print(f'R-squared: {r2}')

accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Overall Accuracy: {accuracy:.2f}')

auc_score = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {auc_score:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
print('\nConfusion Matrix:')
print(cm)

# Predicted labels distribution
from collections import Counter
class_counts = Counter(y_pred_binary)
print('\nPredicted Labels Distribution:')
print(class_counts)

# Results DataFrame
results_df = pd.DataFrame({
    'Image_Code': image_codes_test,  
    'Actual': y_test,    
    'Predicted': y_pred_binary,  
    'Predicted_percent': y_pred_percent
})


results_df.to_csv("Results/MODEL2/CV/cot_cnn_predictions.txt", sep='\t', index=False)

print('saved results')



plt.figure(figsize=(14, 6))

# Define a larger font size
axis_font_size = 14
legend_font_size = 12
title_font_size = 16

# Plot F1 Score
plt.subplot(1, 2, 1)
plt.plot(history.history['f1_score'], label="Training F1 Score", color="#a4041c")  # Custom color
plt.plot(history.history['val_f1_score'], label="Validation F1 Score", color="#4974a5")  # Custom color
plt.xlabel("Epochs", color="black", fontsize=axis_font_size)
plt.ylabel("F1 Score", color="black", fontsize=axis_font_size)
plt.title("F1 Score Over Epochs", color="black", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)
#plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training Loss", color="#a4041c")  # Custom color
plt.plot(history.history['val_loss'], label="Validation Loss", color="#4974a5")  # Custom color
plt.xlabel("Epochs", color="black", fontsize=axis_font_size)
plt.ylabel("Loss", color="black", fontsize=axis_font_size)
plt.title("Loss Over Epochs", color="black", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)
#plt.grid(True)

# Save the combined plot
plt.tight_layout()
plt.savefig("Results/MODEL2/CV/validation_plots.png")
plt.show()
