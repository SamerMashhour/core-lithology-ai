# 📁 File: notebooks/4_cnn_classifier.ipynb
# Description: CNN classifier for core images using TensorFlow/Keras

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

# Load image labels
labels_df = pd.read_csv('../data/geochemistry.csv')
labels_df.set_index('SampleID', inplace=True)

# Load and resize images
image_dir = '../data/sample_core_images'
X, y = [], []
img_size = (64, 64)

for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        sample_id = filename.replace('.png', '')
        label = labels_df.loc[sample_id]['Lithology']
        img = Image.open(os.path.join(image_dir, filename)).resize(img_size)
        X.append(np.array(img) / 255.0)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
label_map = {"Mafic": 0, "Ultramafic": 1, "Felsic": 2}
y_encoded = np.array([label_map[label] for label in y])
y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Classification Accuracy')
plt.grid(True)
plt.show()
