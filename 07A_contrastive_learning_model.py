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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt

import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

print('All packages imported successfully :)')


print(tf.__version__)
print(keras.__version__)
print(keras_cv.__version__)
print('All run ready for datas')



print(' ----')
tf.keras.backend.clear_session()
tf.config.list_physical_devices('GPU')
devices = tf.config.experimental.list_physical_devices('GPU')
print(devices)
for gpu in devices:
  tf.config.experimental.set_memory_growth(gpu, True)
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 16 
EPOCHS = 200#
CROPS_TO = 224
SEED = 26
DIM = (224, 224)
LR = 0.0001
WEIGHT_DECAY = 0.0005


# Load features

# Load features
#feature_file = os.path.join('Data', 'Kenya_Features.csv')
feature_file = pd.read_csv('/../../../mnt/ext_drive/full_features_10km_27_05.csv')

#features = pd.read_csv(feature_file) #, sheet_name='Kenya_Sarah')
print((feature_file.columns))
features = feature_file.drop_duplicates(subset='filenames')
print(features.shape)
print('----')



# Load images 
image_file = os.path.join('Data', 'Arrays', 'Kenya_array_augmented.npy')
images = np.load(image_file, allow_pickle=True)
images = images.item()




df = pd.read_csv(os.path.join('Data', 'clustered_geopoints.txt'), sep = '\t')
print(len(df))
print(df.columns)
df = df.merge(features, on = 'filenames', how= 'left')
print(df)
print(len(df))


#labels = df['avg_wealthscore'].values
df['avg_roofcat'] = df['avg_roofcat'].replace(0.5, 0)
labels = df['avg_roofcat'].values
image_codes = df['filenames'].astype(str).values #imagecodes

#Remove nan
non_nan = ~np.isnan(labels)
labels = labels[non_nan]
image_codes = image_codes[non_nan]

print(df['avg_roofcat'].value_counts())


print(len(image_codes))
print(len(labels))
print(df.columns)
print(features.columns)






X_images = np.array([images[code] for code in image_codes if code in images])
#X_features = df_merge.loc[:, df_merge.columns.str.startswith('feature_')].values
X_features = df.loc[:, df.columns.str.startswith('feature_')].values.astype('float32')

# Labels should be in the 'labels' variable
# labels = labels


X_images = X_images.astype('float32') / 255.0  # Normalize images
labels = labels.astype('int32')

print('X images')
print(len(X_images))
print('labels: ')
print(len(labels))
print('Features')
print(len(X_features))


print(X_images.shape)
print(X_features.shape)
print(labels.shape)

print('------------')



unique, frequency = np.unique(labels,
                              return_counts = True)
# print unique values array
print("Unique Values:",
      unique)

# print frequency array
print("Frequency Values:",
      frequency)

print('------------')


X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test = train_test_split(
    X_images, X_features, labels, test_size=0.2, random_state=42)


print(X_train_images.shape)
print(X_test_images.shape)

print(X_train_features.shape)
print(X_test_features.shape)

print(y_train.shape)
print(y_test.shape)

print(len(labels))


print('labels check: ')
print("Labels (first 10):", labels[:10])
print("Unique Labels:", np.unique(labels))
print("Labels Shape:", labels.shape)




NUM_CLASSES = 2
BATCH_SIZE = 16
INPUT_DIM_FEATURES = X_features.shape[1]
print(INPUT_DIM_FEATURES)



def cosine_similarity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred))


# Preprocess function
def preprocess_data(images, features, labels):

    labels = tf.one_hot(labels, NUM_CLASSES)
    return {"image_input_unique": images, "feature_input_unique": features}, labels


def preprocess_data(images, features, labels):
    # For binary classification, labels should just be 0 or 1, no need for one-hot encoding
    labels = tf.cast(labels, tf.float32)  # Cast to float32 for compatibility
    print(images.shape)
    print(features.shape)
    print(labels.shape)
    return {"image_input_unique": images, "feature_input_unique": features}, labels


def preprocess_data(images, features, labels):
    labels = tf.cast(labels, tf.float32)
    return (images, features), labels  # Return inputs as a tuple



# Create TensorFlow datasets from both images and features
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_images, X_train_features), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_images, X_test_features), y_test))



train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


print('Train dataset')
print(train_dataset)






from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

# Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weights_dict)





# Constants
NUM_CLASSES = 2  # For binary classification

# Define image input (shape: 224x224x3)
image_input = keras.Input(shape=(224, 224, 3), name="image_input_unique")

# Process images using a CNN
x_image = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
x_image = BatchNormalization()(x_image)
x_image = Conv2D(32, (3, 3), activation='relu', padding='same')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)

x_image = Conv2D(64, (3, 3), activation='relu', padding='same')(x_image)
x_image = BatchNormalization()(x_image)
x_image = Conv2D(64, (3, 3), activation='relu', padding='same')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)

x_image = Conv2D(128, (3, 3), activation='relu', padding='same')(x_image)
x_image = BatchNormalization()(x_image)
x_image = Conv2D(128, (3, 3), activation='relu', padding='same')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)

x_image = Conv2D(256, (3, 3), activation='relu', padding='same')(x_image)
x_image = BatchNormalization()(x_image)
x_image = Conv2D(256, (3, 3), activation='relu', padding='same')(x_image)
x_image = MaxPooling2D((2, 2))(x_image)


x_image = keras.layers.Flatten()(x_image)
x_image = keras.layers.Dense(128, activation='relu')(x_image)

# Define feature input 
feature_input = keras.Input(shape=(1024,), name="feature_input_unique")
#feature_input = keras.Input(shape=(INPUT_DIM_FEATURES,), name="feature_input_unique")

# Combine the processed image features with the additional weightings (features)
combined = keras.layers.concatenate([x_image, feature_input])

# Add dense layers after combining both inputs
combined = keras.layers.Dense(64, activation='relu')(combined)
combined = keras.layers.Dropout(0.5)(combined)  
combined = keras.layers.Dense(32, activation='relu')(combined)
combined = keras.layers.Dropout(0.5)(combined)


# Output layer for binary classification
output = keras.layers.Dense(1, activation='sigmoid')(combined)

# Define the model
model = keras.Model(inputs=[image_input, feature_input], outputs=output)


# Compile the model
model.compile(
    loss='binary_crossentropy',  
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics= ['accuracy']
    #[metrics=[cosine_similarity]] #metrics=['AUC']
)



# Display model summary
model.summary()








log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)




model.fit(
    train_dataset,                # Training data
    #validation_data=test_dataset,  # Validation data
    epochs=EPOCHS,                    # Number of epochs
    verbose=1,                       # Show progress
    class_weight = class_weights_dict,
    callbacks=[tensorboard_callback]               
    #callbacks=[print_shape_callback]  # Pass the custom callback
)


import subprocess
#subprocess.run(["tensorboard", "--logdir", log_dir, "--port", "6006"])
#print(f"TensorBoard is running at http://localhost:6006")


model.evaluate(test_dataset)



from sklearn.metrics import mean_absolute_error

y_pred = model.predict(test_dataset)
y_pred = (y_pred > 0.45).astype(int)
#y_pred = np.argmax(y_pred, axis=1)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')



from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#print(y_pred)


cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix: ')
print(cm)


print('Predicted Labels: ')

class_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
print(Counter(class_counts))




# Roc CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
image_path = os.path.join('Data', 'Results', 'contrast_roc.png')



y_pred_prob = model.predict(test_dataset)
y_pred_prob = y_pred_prob.ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)



plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Dashed diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")


plt.savefig(image_path, format='png')


# I want to increase recall in class 0