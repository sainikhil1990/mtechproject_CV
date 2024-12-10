import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Parameters
image_folder = '/home/eiiv-nn1-l3t04/conv/image_file/dataset/normal'
image_size = (64, 64)  # Resize images to 64x64

def load_images(image_folder, image_size):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Load and preprocess images
images = load_images(image_folder, image_size)
images_flattened = images.reshape(images.shape[0], -1)  # Flatten images

# Normalize the data
scaler = StandardScaler()
images_normalized = scaler.fit_transform(images_flattened)

# Train One-Class SVM
oc_svm = OneClassSVM(gamma='auto', nu=0.1)  # nu is the outlier fraction
oc_svm.fit(images_normalized)

def detect_anomaly_oc_svm(image_path):
    img = load_img(image_path, target_size=image_size, color_mode='grayscale')
    img_array = img_to_array(img).reshape(1, -1)
    img_normalized = scaler.transform(img_array)
    prediction = oc_svm.predict(img_normalized)
    return prediction[0] == -1

# Test
test_image_path = '/home/eiiv-nn1-l3t04/Project/dataset/testing/cubes.jpg.23im33f2.ingestion-6797d84bf-dn6tb.jpg.41237u4o.ingestion-64f85c558f-6hht2.jpg'
is_anomaly = detect_anomaly_oc_svm(test_image_path)
print("Anomaly detected:", is_anomaly)