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

from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import Recall
from sklearn.utils import class_weight
#from tensorflow.keras import backend as K


from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model


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
BATCH_SIZE = 8 
EPOCHS = 200
CROPS_TO = 224
SEED = 42
DIM = (224, 224)
LR = 0.0001
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2




#tf.random.set_seed(42)
#np.random.seed(42)



# Load features

# Load features
#feature_file = os.path.join('Data', 'Kenya_Features.csv')
#feature_file = pd.read_csv('/../../../mnt/ext_drive/full_features_10km_27_05.csv')
feature_file =pd.read_csv('Data/Feature_weights/image_features_with_filenames.csv')
feature_file = feature_file.rename(columns={'filename': 'filenames'})
#features = pd.read_csv(feature_file) #, sheet_name='Kenya_Sarah')
features = feature_file.drop_duplicates(subset='filenames')
features = feature_file
print('----')

# Load images 
image_file = os.path.join('Data', 'Arrays', 'Kenya_array_augmented.npy')
images = np.load(image_file, allow_pickle=True)
images = images.item()





df = pd.read_csv(os.path.join('Data', 'clustered_geopoints.txt'), sep = '\t')
print(len(df))
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



#X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test, image_codes_train, image_codes_test = train_test_split(
#    X_images, X_features, labels, image_codes, test_size=0.2, random_state=42)



# Test with 70 15 15 

X_train_val_images, X_test_images, X_train_val_features, X_test_features, y_train_val, y_test, image_codes_train_val, image_codes_test = train_test_split(
    X_images, X_features, labels, image_codes, test_size=0.15, random_state=42)


X_train_images, X_val_images, X_train_features, X_val_features, y_train, y_val = train_test_split(
    X_train_val_images, X_train_val_features, y_train_val, test_size=0.1765, random_state=42)  










print(X_train_images.shape)
print(X_test_images.shape)
print(X_val_images.shape)

print(X_train_features.shape)
print(X_test_features.shape)
print(X_val_features.shape)



print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

print(len(labels))


print('labels check: ')
print("Labels (first 10):", labels[:10])
print("Unique Labels:", np.unique(labels))
print("Labels Shape:", labels.shape)






def cosine_similarity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred))



def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)    


def recall_class_0(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = tf.round(y_pred)

    # Filter to only consider class 0
    true_positives = K.sum(K.cast((y_true == 0) & (y_pred == 0), 'float'))
    possible_positives = K.sum(K.cast(y_true == 0, 'float'))

    recall = true_positives / (possible_positives + K.epsilon())
    return recall    





# Preprocess function
def preprocess_data(images, features, labels):
    labels = tf.cast(labels, tf.float32)
    return (images, features), labels  # Return inputs as a tuple



# Create TensorFlow datasets from both images and features
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_images, X_train_features), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_images, X_test_features), y_test))
val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_images, X_val_features), y_val))











train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


val_dataset = val_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)



# Class weights

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict[0] *= 2


#class_weights_dict = {0: 2, 1: 1}
print("Class Weights:", class_weights_dict)



INPUT_DIM_FEATURES = X_features.shape[1]





#vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the VGG16 layers if you don’t want to retrain them
for layer in vgg_base.layers[-6:]:
    layer.trainable = True

# Process images with VGG16
x_image = vgg_base.output
x_image = Flatten()(x_image)
#x_image = Dense(128, activation='relu')(x_image)
x_image = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x_image) 

# Define feature input (512 weightings)
feature_input = keras.Input(shape=(INPUT_DIM_FEATURES,), name="feature_input_unique")

# Combine the processed image features with the additional weightings (features)
combined = Concatenate()([x_image, feature_input])



combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)  
combined = Dropout(0.5)(combined)
combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined) 
combined = Dropout(0.5)(combined)





# Output layer for binary classification
output = Dense(1, activation='sigmoid')(combined)

# Define the model with VGG16 as the image input model
model = Model(inputs=[vgg_base.input, feature_input], outputs=output)

#import tensorflow_addons as tfa


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)

#change lr_schedule patience




# Compile the model
model.compile(
    loss='binary_crossentropy',   # This one is good
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # This one is good 
    #metrics=[f1_score]
    metrics=[f1_score, Recall(class_id=0)] 
)

# Display model summary
model.summary()



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    class_weight=class_weights_dict,
    callbacks=[tensorboard_callback, lr_schedule, early_stopping]  
)
#
    #callbacks=[print_shape_callback]  


print("\nEvaluating on the test dataset:")
test_loss, test_f1_score, test_recall_class_0  = model.evaluate(test_dataset)


from sklearn.metrics import mean_absolute_error

y_pred = model.predict(test_dataset)
#y_pred = (y_pred > 0.5).astype(int)
y_pred_percent = y_pred
y_pred = (y_pred > 0.65).astype(int)
#y_pred = np.argmax(y_pred, axis=1)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')



from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {accuracy:.2f}')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix: ')
print(cm)


print('Predicted Labels: ')

class_counts = dict(zip(*np.unique(y_pred, return_counts=True)))
print(Counter(class_counts))


results_df = pd.DataFrame({
     'Image_Code': image_codes_test,  
    'Actual': y_test,    
    'Predicted': y_pred.flatten(),  
    'Predicted_percent': y_pred_percent.flatten()
})


model.save("Results/MODEL3/full_model.keras")
model.save_weights("Results/MODEL3/model_weights.weights.h5")


#### Save predictions




# Save to a CSV file

results_df.to_csv("Results/MODEL3/cot_cnn_predictions.txt", sep='\t', index=False)




print('saved results')


#################################################################

plt.figure(figsize=(14, 6))

# Plot F1 Score
plt.subplot(1, 2, 1)
plt.plot(history.history['f1_score'], label="Training F1 Score", color="#a4041c")  # Custom color
plt.plot(history.history['val_f1_score'], label="Validation F1 Score", color="#4974a5")  # Custom color
plt.xlabel("Epochs", color="black")
plt.ylabel("F1 Score", color="black")
plt.title("F1 Score Over Epochs", color="black")
plt.legend()
#plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training Loss", color="#a4041c")  # Custom color
plt.plot(history.history['val_loss'], label="Validation Loss", color="#4974a5")  # Custom color
plt.xlabel("Epochs", color="black")
plt.ylabel("Loss", color="black")
plt.title("Loss Over Epochs", color="black")
plt.legend()
#plt.grid(True)

# Save the combined plot
plt.tight_layout()
plt.savefig("Results/MODEL3/validation_plots.png")
plt.show()


### AUC PLOT 


fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color="#a4041c", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})") 
plt.plot([0, 1], [0, 1], color="#4974a5", lw=2, linestyle="--", label="Random") 

# Set labels and title
plt.xlabel("False Positive Rate", color="black")
plt.ylabel("True Positive Rate", color="black")
plt.title("ROC Curve", color="black")
plt.legend()
plt.grid(True)

# Save the plot

plt.savefig("Results/MODEL3/roc_curve.png")
plt.show()













###################################################


import subprocess
subprocess.run(["tensorboard", "--logdir", log_dir, "--port", "6006"])
print(f"TensorBoard is running at http://localhost:6006")