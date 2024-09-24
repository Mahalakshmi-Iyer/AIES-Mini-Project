import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Normalize the image data to the range [0, 1]
X_train = X_train.astype('float32') / 255.0

# Ensure the labels are one-hot encoded (6 classes for your categories)
y_train = to_categorical(y_train, num_classes=6)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Load the saved model
model = load_model('waste_classification_model.h5')

# Evaluate the model (ensure y_val is one-hot encoded as well)
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Load the saved history from the file
history = np.load('history.npy', allow_pickle=True).item()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
