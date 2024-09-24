import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define image size
IMG_SIZE = 128  # Ensure this matches the size used in preprocess.py

# Check if preprocessed files exist, otherwise run preprocess.py
if not os.path.exists('X_train.npy') or not os.path.exists('y_train.npy'):
    print("Preprocessed data not found. Running preprocess.py...")
    import preprocess  # Assuming preprocess.py generates X_train.npy and y_train.npy

# Load the preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Normalize the image data to the range [0, 1]
X_train = X_train.astype('float32') / 255.0

# One-hot encode the labels (assuming 6 classes)
y_train = to_categorical(y_train, num_classes=6)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4th Convolutional Layer (optional)
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Output Layer
model.add(Dense(6, activation='softmax'))  # 6 classes for waste categories

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(
    X_val, y_val), epochs=20, batch_size=32)

# Save the trained model
model.save('waste_classification_model.h5')

np.save('history.npy', history.history)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

print("Model training complete and saved as 'waste_classification_model.h5'.")
