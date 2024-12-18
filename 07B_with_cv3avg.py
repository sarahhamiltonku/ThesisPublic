import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (confusion_matrix, mean_absolute_error, accuracy_score, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
from collections import Counter

# Define parameters
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 8
EPOCHS = 200
LR = 0.0001

# Load features
feature_file = pd.read_csv('Data/Feature_weights/image_features_with_filenames.csv')
feature_file = feature_file.rename(columns={'filename': 'filenames'})
features = feature_file.drop_duplicates(subset='filenames')

# Load images
image_file = os.path.join('Data', 'Arrays', 'Kenya_array_augmented.npy')
images = np.load(image_file, allow_pickle=True).item()

# Load main DataFrame
df = pd.read_csv(os.path.join('Data', 'clustered_geopoints.txt'), sep='\t')
df = df.merge(features, on='filenames', how='left')

# Process labels and image codes
df['avg_roofcat'] = df['avg_roofcat'].replace(0.5, 0)
labels = df['avg_roofcat'].values
image_codes = df['filenames'].astype(str).values

# Remove NaNs
non_nan = ~np.isnan(labels)
labels = labels[non_nan]
image_codes = image_codes[non_nan]

# Process features and images
X_images = np.array([images[code] for code in image_codes if code in images]).astype('float32') / 255.0
X_features = df.loc[:, df.columns.str.startswith('feature_')].values.astype('float32')

# Split into training and test datasets
X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test, image_codes_train, image_codes_test = train_test_split(
    X_images, X_features, labels, image_codes, test_size=0.2, random_state=3
)

# Define preprocessing
def preprocess_data(images, features, labels):
    labels = tf.cast(labels, tf.float32)
    return (images, features), labels

# Define custom F1 metric
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())



test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_images, X_test_features), y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# Initialize KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=3)

# Cross-validation
cv_results = []
fold_confusion_matrices = []
all_y_true, all_y_pred = [], []
all_y_pred_percent = []
cv_histories = []
cv_models = []

for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_images, labels), 1):
    print(f"Training fold {fold_no}...")
    train_images, val_images = X_images[train_idx], X_images[val_idx]
    train_features, val_features = X_features[train_idx], X_features[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_images, train_features), train_labels))
    train_dataset = train_dataset.batch(BATCH_SIZE).map(
        lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
        num_parallel_calls=AUTO).prefetch(AUTO)

    val_dataset = tf.data.Dataset.from_tensor_slices(((val_images, val_features), val_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE).map(
        lambda x_img_feat, y: preprocess_data(x_img_feat[0], x_img_feat[1], y),
        num_parallel_calls=AUTO).prefetch(AUTO)

    # Build model
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_base.layers[-6:]:
        layer.trainable = True

    x_image = Flatten()(vgg_base.output)
    x_image = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x_image)
    feature_input = Input(shape=(X_features.shape[1],), name="feature_input")
    combined = Concatenate()([x_image, feature_input])
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[vgg_base.input, feature_input], outputs=output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        metrics=[f1_score]
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    tensorboard_callback = TensorBoard(log_dir=f"logs/fold_{fold_no}/")

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stopping, lr_schedule, tensorboard_callback]
    )
    cv_histories.append(history)
    cv_models.append(model)


    # Evaluate fold
    val_loss, val_f1_score = model.evaluate(val_dataset, verbose=0)
    print(f"Fold {fold_no} - Loss: {val_loss}, F1 Score: {val_f1_score}")

    # Predict fold
    y_pred = model.predict(val_dataset).flatten()
    y_pred_binary = (y_pred > 0.75).astype(int)
    fold_cm = confusion_matrix(val_labels, y_pred_binary)

    cv_results.append({'fold': fold_no, 'val_loss': val_loss, 'f1_score': val_f1_score})
    fold_confusion_matrices.append(fold_cm)
    all_y_true.extend(val_labels)
    all_y_pred.extend(y_pred_binary)
    all_y_pred_percent.extend(y_pred)

# Aggregate results
# Initialize accumulators for averaging metrics
avg_train_f1_scores = []
avg_val_f1_scores = []
avg_train_losses = []
avg_val_losses = []

# Loop through each fold's history and accumulate metrics
for fold_history in cv_histories:  # Assuming cv_histories is a list of Keras history objects
    avg_train_f1_scores.append(fold_history.history['f1_score'])
    avg_val_f1_scores.append(fold_history.history['val_f1_score'])
    avg_train_losses.append(fold_history.history['loss'])
    avg_val_losses.append(fold_history.history['val_loss'])

# Compute average metrics across folds
epochs = range(1, len(avg_train_f1_scores[0]) + 1)  # Assuming all folds have same number of epochs
mean_train_f1 = np.mean(avg_train_f1_scores, axis=0)
mean_val_f1 = np.mean(avg_val_f1_scores, axis=0)
mean_train_loss = np.mean(avg_train_losses, axis=0)
mean_val_loss = np.mean(avg_val_losses, axis=0)

# Plot average F1 Score and Loss
plt.figure(figsize=(14, 6))

# Define font sizes
axis_font_size = 14
legend_font_size = 14
title_font_size = 18

# Plot Average F1 Score
plt.subplot(1, 2, 1)
plt.plot(epochs, mean_train_f1, label="Training F1 Score", color="#a4041c")
plt.plot(epochs, mean_val_f1, label="Validation F1 Score", color="#4974a5")
plt.xlabel("Epochs", color="black", fontsize=axis_font_size)
plt.ylabel("F1 Score", color="black", fontsize=axis_font_size)
plt.title("F1 Score Over Epochs", color="black", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)

# Plot Average Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, mean_train_loss, label="Training Loss", color="#a4041c")
plt.plot(epochs, mean_val_loss, label="Validation Loss", color="#4974a5")
plt.xlabel("Epochs", color="black", fontsize=axis_font_size)
plt.ylabel("Loss", color="black", fontsize=axis_font_size)
plt.title("Loss Over Epochs", color="black", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)

# Save the combined plot
plt.tight_layout()
plt.savefig("Results/MODEL3/CV/avg/average_validation_plots.png")
plt.show()


y_pred = model.predict(test_dataset).flatten()
y_pred_percent = y_pred  # Raw Probs
y_pred_binary = (y_pred > 0.75).astype(int)  

results_df = pd.DataFrame({
    'Image_Code': image_codes_test,  
    'Actual': y_test,       
    'Predicted': y_pred_binary,        
    'Predicted_percent': y_pred_percent    
})

# Save the DataFrame to a CSV file
results_file = "Results/MODEL3/CV/avg/cot_cnn_predictions.txt"
results_df.to_csv(results_file, sep='\t', index=False)
print(f"Results saved to {results_file}")





average_model = tf.keras.models.clone_model(cv_models[0])  # Clone structure from the first model
average_weights = [np.mean([model.get_weights()[i] for model in cv_models], axis=0)
                   for i in range(len(cv_models[0].get_weights()))]
average_model.set_weights(average_weights)

# Use the averaged model for predictions
final_test_preds = average_model.predict(test_dataset).flatten()
final_test_binary = (final_test_preds > 0.75).astype(int)



# Metrics calculation
mae = mean_absolute_error(y_test, final_test_binary)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, final_test_binary)
print(f'R-squared: {r2}')

accuracy = accuracy_score(y_test, final_test_binary)
print(f'Overall Accuracy: {accuracy:.2f}')

auc_score = roc_auc_score(y_test, final_test_binary)
print(f'ROC AUC Score: {auc_score:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, final_test_binary))

# Confusion Matrix
cm = confusion_matrix(y_test, final_test_binary)
print('\nConfusion Matrix:')
print(cm)
