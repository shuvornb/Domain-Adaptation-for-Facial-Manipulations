import numpy as np
import pandas as pd
from skimage import io
import cv2
import keras
import tensorflow as tf
import time
import sys

# Start the timer
start_time = time.time()

# Constants
img_rows, img_cols = 128, 128
num_classes = 2

# Load data from CSV files
train_data = pd.read_csv("csv_files/train_data.csv")
test_data_source = pd.read_csv("csv_files/test_data_source.csv")
test_data_target = pd.read_csv("csv_files/test_data_target.csv")

# Define mappings
category_mapping = {"fake": 0, "real": 1, "unlabeled": -1}
domain_mapping = {sys.argv[1]: 0, sys.argv[2]: 1}

# Map categories and domains
train_data["Category"] = train_data["Category"].map(category_mapping)
test_data_source["Category"] = test_data_source["Category"].map(category_mapping)
test_data_target["Category"] = test_data_target["Category"].map(category_mapping)

train_data["Domain"] = train_data["Domain"].map(domain_mapping)
test_data_source["Domain"] = test_data_source["Domain"].map(domain_mapping)
test_data_target["Domain"] = test_data_target["Domain"].map(domain_mapping)

# Process image data
def process_image_data(data):
    x = np.zeros((len(data), img_rows, img_cols, 3))
    y_c = np.zeros((len(data), 1))
    y_d = np.zeros((len(data), 1))
    not_found = 0

    file_count = 0
    total_files = len(data)

    for index, row in data.iterrows():
        full_path = str(row['Full_Path']).rstrip()

        file_count += 1
        progress = (file_count / total_files) * 100
        print(f"Processing Image: {progress:.2f}% complete", end="\r", flush=True)

        try:
            image = io.imread(full_path)
        except FileNotFoundError:
            not_found += 1
            continue
        x[index] = cv2.resize(image, (img_rows, img_cols))
        y_c[index] = row['Category']
        y_d[index] = row['Domain']
    return x, y_c, y_d, not_found

# Process train data
x_train, y_train_c, y_train_d, not_found = process_image_data(train_data)

# Process source test data
x_test_source, y_test_c_source, y_test_d_source, not_found_test_source = process_image_data(test_data_source)

# Process target test data
x_test_target, y_test_c_target, y_test_d_target, not_found_test_target = process_image_data(test_data_target)

# Convert labels to categorical
y_train_c = keras.utils.to_categorical(y_train_c, num_classes)
y_train_d = keras.utils.to_categorical(y_train_d, 2)
y_test_c_source = keras.utils.to_categorical(y_test_c_source, num_classes)
y_test_d_source = keras.utils.to_categorical(y_test_d_source, 2)
y_test_c_target = keras.utils.to_categorical(y_test_c_target, num_classes)
y_test_d_target = keras.utils.to_categorical(y_test_d_target, 2)

# Preprocess input images
x_train = tf.keras.applications.xception.preprocess_input(x_train)
x_test_source = tf.keras.applications.xception.preprocess_input(x_test_source)
x_test_target = tf.keras.applications.xception.preprocess_input(x_test_target)

print("Total Training Image found:", len(train_data)- not_found)
print("Total Source Test Image found:", len(test_data_source) - not_found_test_source)
print("Total Target Test Image found:", len(test_data_target) - not_found_test_target)

# Save processed data to disk
np.save("processed_data/x_train.npy", x_train)
np.save("processed_data/y_train_c.npy", y_train_c)
np.save("processed_data/y_train_d.npy", y_train_d)
np.save("processed_data/x_test_source.npy", x_test_source)
np.save("processed_data/x_test_target.npy", x_test_target)
np.save("processed_data/y_test_c_source.npy", y_test_c_source)
np.save("processed_data/y_test_d_source.npy", y_test_d_source)
np.save("processed_data/y_test_c_target.npy", y_test_c_target)
np.save("processed_data/y_test_d_target.npy", y_test_d_target)

# total time elapsed for data processing
total_time = time.time() - start_time
print("Total Time:", total_time, "seconds")
