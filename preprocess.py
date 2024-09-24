import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to your dataset folder
dataset_dir = "./garbage_classification_dataset"

# Categories based on folder names
categories = ['paper', 'plastic', 'metal', 'scrap', 'cardboard', 'glass']

# Image size (you can adjust this)
IMG_SIZE = 128

# Data lists
data = []
labels = []

# Loop through each category folder
for category in categories:
    folder_path = os.path.join(dataset_dir, category)
    class_num = categories.index(category)  # Label each category
    for img in os.listdir(folder_path):
        try:
            # Read the image
            img_path = os.path.join(folder_path, img)
            img_array = cv2.imread(img_path)

            # Handle cases where the image is not loaded properly
            if img_array is None:
                print(f"Failed to load image {img_path}")
                continue

            # Resize the image
            resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # Append the image and its label
            data.append(resized_img)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Convert to numpy arrays and normalize
data = np.array(data).astype('float32') / 255.0
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Save the preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Data preprocessing complete. Preprocessed data saved as .npy files.")
