
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




# Normalise images
images = {key: np.array(img).astype('float32') / 255.0 for key, img in images.items()}
X = np.array([images[code] for code in image_codes if code in images])


X_images = np.array([images[code] for code in image_codes if code in images])
X_images = X_images.astype('float32') / 255.0  # Normalize images

print(f"Image data shape: {X.shape}")
print(f"Labels shape: {labels.shape}")

print(len(X), len(labels), len(image_codes))


#X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)




X_train_val_images, X_test_images, y_train_val, y_test, image_codes_train_val, image_codes_test = train_test_split(
    X_images, labels, image_codes, test_size=0.15, random_state=9)


X_train_images, X_val_images, y_train, y_val = train_test_split(
    X_train_val_images, y_train_val, test_size=0.1765, random_state=9)  




print(X_train_images.shape)
print(X_test_images.shape)
print(X_val_images.shape)


print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

print(len(labels))


print('labels check: ')
print("Labels (first 10):", labels[:10])
print("Unique Labels:", np.unique(labels))
print("Labels Shape:", labels.shape)







# Flat arrays
X_train_flat = X_train_images.reshape(X_train_images.shape[0], -1)
X_test_flat = X_test_images.reshape(X_test_images.shape[0], -1)
X_val_flat = X_val_images.reshape(X_val_images.shape[0], -1)

def preprocess_data(images, labels):
    labels = tf.cast(labels, tf.int32) 
    images = tf.image.resize(images, (224, 224)) 
    #labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels


def preprocess_data(images, labels):
    labels = tf.cast(labels, tf.float32)
    return images, labels 









# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_images, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_images, y_test))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_images, y_val))

# Apply preprocessing, batching, and prefetching
train_dataset = train_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


test_dataset = test_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(
    lambda x, y: preprocess_data(x, y),
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

backbone = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")

for _, label in train_dataset.take(5):
    print("Label:", label.numpy(), "Shape:", label.numpy().shape)









# Add class weights


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
#Convert class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict[0] *= 2

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


test_loss, test_f1_score, test_recall_class_0 = model.evaluate(test_dataset)
#, test_accuracy_score



from sklearn.metrics import mean_absolute_error

y_pred = model.predict(test_dataset)
y_pred_percent = y_pred
y_pred = (model.predict(test_dataset) > 0.5).astype(int).flatten()

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')




print(y_pred.shape)
print(y_test.shape)


accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {accuracy:.2f}')


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')




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



print('saved results')


#### Save predictions



# Save to a CSV file

results_df.to_csv("Results/MODEL2/trans_cnn_predictions.txt", sep='\t', index=False)



#######################################


# Plots 


plt.figure(figsize=(14, 6))

# Plot F1 Score
plt.subplot(1, 2, 1)
plt.plot(history.history['f1_score'], label="Training F1 Score", color="#a4041c")  
plt.plot(history.history['val_f1_score'], label="Validation F1 Score", color="#4974a5") 
plt.xlabel("Epochs", color="black")
plt.ylabel("F1 Score", color="black")
plt.title("F1 Over Epochs", color="black")
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
plt.savefig("Results/MODEL2/validation_plots_model2.png")
plt.show()





### AUC PLOT 


fpr, tpr, _ = roc_curve(y_test, y_pred_percent)
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

plt.savefig("Results/MODEL2/roc_curveMODEL2a.png")
plt.show()





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

plt.savefig("Results/MODEL2/roc_curveMODEL2b.png")
plt.show()










"""

plt.figure(figsize=(14, 6))

# Plot F1 Score
plt.subplot(1, 2, 1)
plt.plot(history.history['f1_score'], label="Training F1 Score")
plt.plot(history.history['val_f1_score'], label="Validation F1 Score")  # Validation F1 score
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Epochs")
plt.legend()
plt.savefig("Results/MODEL2/ValidationF1.png")
plt.tight_layout()
plt.close() 




# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training Accuracy Score")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy Score")  # Validation F1 score
plt.xlabel("Epochs")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score Over Epochs")
plt.legend()
plt.savefig("Results/MODEL2/ValidationACC.png")
plt.tight_layout()
plt.close() 




# Loss Plot


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")  # Validation loss
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("Results/MODEL2/loss_over_epochs.png")
plt.tight_layout()
plt.close() 



"""





