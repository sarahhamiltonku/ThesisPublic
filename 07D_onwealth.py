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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report




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
SEED = 9
DIM = (224, 224)
LR = 0.0001
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 5
BATCH_SIZE = 8


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
#df['avg_roofcat'] = df['avg_roofcat'].replace(0.5, 0)
labels = df['avg_wealthscore'].values
image_codes = df['filenames'].astype(str).values #imagecodes

#Remove nan
non_nan = ~np.isnan(labels)
labels = labels[non_nan]
labels = labels - 1
image_codes = image_codes[non_nan]
df = df.dropna(subset=['avg_wealthscore'])

print(df['avg_wealthscore'].value_counts())


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



# Functions




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









# Split into Train and Test





X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test, image_codes_train, image_codes_test = train_test_split(
    X_images, X_features, labels, image_codes, test_size=0.2, random_state=3)



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


# Start fold 

kfold = KFold(n_splits=1, shuffle=True, random_state=3)




test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_images, X_test_features), y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

#class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# Convert class weights to a dictionary
#class_weights_dict = dict(enumerate(class_weights))
#class_weights_dict[0] *= 2

class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(labels), 
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))





fold_no = 1
cv_results = []
for train_idx, val_idx in kfold.split(X_train_features, y_train):
    print(f"Training fold {fold_no}...")

    train_images = X_images[train_idx]
    train_features = X_features[train_idx]
    train_labels = labels[train_idx]

    val_images = X_images[val_idx]
    val_features = X_features[val_idx]
    val_labels = labels[val_idx]



    train_dataset = tf.data.Dataset.from_tensor_slices(((train_images, train_features), train_labels))
    train_dataset = train_dataset.batch(BATCH_SIZE).map(
        lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(((val_images, val_features), val_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE).map(
        lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)




    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_base.layers[-6:]:
        layer.trainable = True

    x_image = vgg_base.output
    x_image = Flatten()(x_image)
    x_image = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x_image)

    feature_input = tf.keras.Input(shape=(X_features.shape[1],), name="feature_input_unique")
    combined = Concatenate()([x_image, feature_input])
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(NUM_CLASSES, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs=[vgg_base.input, feature_input], outputs=output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #metrics=[tf.keras.metrics.Recall(class_id=0), f1_score]
        metrics = ['accuracy']
    )



    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
    log_dir = f"logs/fold_{fold_no}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)




    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weights_dict,
        callbacks=[tensorboard_callback, lr_schedule, early_stopping]
    )


    #val_loss, val_f1_score, val_recall_class_0 = model.evaluate(val_dataset)
    #print(f"Validation results for fold {fold_no}: Loss={val_loss}, F1 Score={val_f1_score}, Recall for Class 0={val_recall_class_0}")
    

    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation results for fold {fold_no}: Loss={val_loss}, Accuracy={val_accuracy}")

    cv_results.append({'fold': fold_no, 'val_loss': val_loss, 'Accuracy': val_accuracy})
    fold_no += 1




print("\nFinal evaluation on the test dataset:")
#test_loss, test_f1_score, test_recall_class_0 = model.evaluate(test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test results: Loss={test_loss}, Accuracy={test_acc}")
print(cv_results)




y_pred = model.predict(test_dataset)
#.flatten()


print(y_pred)
print(y_pred.shape)
y_pred_percent = y_pred  # Raw Probs
#y_pred_binary = (y_pred > 0.75).astype(int)  
#y_pred_class = np.argmax(y_pred, axis=1)

y_pred_class = np.argmax(y_pred, axis=1) 
print("Predicted Classes (y_pred_class):")
print(y_pred_class)




accuracy = accuracy_score(y_test, y_pred_class)
print(classification_report(y_test, y_pred_class))
cm = confusion_matrix(y_test, y_pred_class)
print(cm)

# Metrics calculation
#mae = mean_absolute_error(y_test, y_pred_binary)
#print(f'Mean Absolute Error: {mae}')

#r2 = r2_score(y_test, y_pred_binary)
#print(f'R-squared: {r2}')

#accuracy = accuracy_score(y_test, y_pred_binary)
#print(f'Overall Accuracy: {accuracy:.2f}')

#auc_score = roc_auc_score(y_test, y_pred)
#print(f'ROC AUC Score: {auc_score:.2f}')

#print("\nClassification Report:")
#print(classification_report(y_test, y_pred_binary))

# Confusion Matrix
#cm = confusion_matrix(y_test, y_pred_binary)
#print('\nConfusion Matrix:')
#print(cm)

# Predicted labels distribution
#from collections import Counter
#class_counts = Counter(y_pred_binary)
#print('\nPredicted Labels Distribution:')
#print(class_counts)

# Results DataFrame
results_df = pd.DataFrame({
    'Image_Code': image_codes_test,  
    'Actual': y_test,    
    'Predicted': y_pred_class,  
   # 'Predicted_percent': y_pred_percent
})


results_df.to_csv("Results/MODEL3/wealth/cot_cnn_predictions.txt", sep='\t', index=False)

#print('saved results')#



plt.figure(figsize=(14, 6))

# Define a larger font size
axis_font_size = 14
legend_font_size = 14
title_font_size = 18

# Plot F1 Score
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training Accuracy", color="#a4041c")  # Custom color
plt.plot(history.history['val_accuracy'], label="Validation Accuracy", color="#4974a5")  # Custom color
plt.xlabel("Epochs", color="black", fontsize=axis_font_size)
plt.ylabel("Accuracy", color="black", fontsize=axis_font_size)
plt.title("Accuracy Over Epochs", color="black", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)

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
plt.savefig("Results/MODEL3/wealth/validation_plots.png")
plt.show()



