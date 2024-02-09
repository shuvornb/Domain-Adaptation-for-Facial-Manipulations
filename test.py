import numpy as np
import pandas as pd
from skimage import io
import cv2
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Activation, Lambda
import tensorflow as tf
import time
import keras
from sklearn.metrics import f1_score
import sys
import os

img_rows, img_cols = 128, 128
num_classes = 2
num_labeled_per_class = 110
num_unlabeled = 30
source = sys.argv[1]
target = sys.argv[2]
trial_no = sys.argv[3]
directory = source + "_" + target + "_" + str(trial_no)
x_test_source = np.load("processed_data/x_test_source.npy")
x_test_target = np.load("processed_data/x_test_target.npy")
y_test_c_source = np.load("processed_data/y_test_c_source.npy")
y_test_c_target = np.load("processed_data/y_test_c_target.npy")

# Start the timer
start_time = time.time()

# helper functions
def save_to_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)

def calculate_f1_score(prediction, actual):
    prediction_labels = np.argmax(prediction, axis=1)
    actual_labels = np.argmax(actual, axis=1)
    return f1_score(actual_labels, prediction_labels)

def flip_gradient(x, l=1.0):
    positive_path = tf.stop_gradient(x * (1 + l))
    negative_path = -x * l
    return positive_path + negative_path

def customLoss(num_labeled_per_class, num_classes, num_unlabeled):
    def contrastive_loss(y_true, y_pred, num_labeled_per_class, temperature):
        labeled_fake = y_pred[:num_labeled_per_class]
        labeled_real = y_pred[num_labeled_per_class:num_labeled_per_class*2]
        #unlabeled = y_pred[num_labeled_per_class*2:]

        # Reshape the labeled_fake and labeled_real to match pairs
        labeled_fake_pairs = tf.reshape(labeled_fake, (num_labeled_per_class // 2, 2, -1))
        labeled_real_pairs = tf.reshape(labeled_real, (num_labeled_per_class // 2, 2, -1))

        # Normalize the feature vectors
        labeled_fake_pairs_norm = tf.nn.l2_normalize(labeled_fake_pairs, axis=-1)
        labeled_real_pairs_norm = tf.nn.l2_normalize(labeled_real_pairs, axis=-1)

        # Calculate logits for similarity between positive and negative pairs
        positive_similarity = tf.reduce_sum(labeled_real_pairs_norm[:, 0] * labeled_real_pairs_norm[:, 1], axis=-1)
        negative_similarity = tf.reduce_sum(labeled_fake_pairs_norm * labeled_real_pairs_norm[:, tf.newaxis, :], axis=-1)

        # Scale logits by temperature
        positive_similarity /= temperature
        negative_similarity /= temperature

        # Calculate the contrastive loss using InfoNCE loss formulation
        positive_loss = -tf.math.log(tf.math.exp(positive_similarity) / (tf.math.exp(positive_similarity) + tf.reduce_sum(tf.math.exp(negative_similarity), axis=-1)))
        negative_loss = -tf.reduce_logsumexp(negative_similarity, axis=-1)

        # Combine positive and negative losses
        loss = tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)

        return loss
    
    def maskedloss_function(y_true, y_pred):

        # Calculate supervised loss (loss for labeled data)
        mask = tf.concat([tf.ones(num_labeled_per_class * num_classes, dtype=tf.bool),
                           tf.zeros(num_unlabeled, dtype=tf.bool)], axis=0)
        supervised_loss = K.categorical_crossentropy(tf.boolean_mask(y_true, mask),
                                                      tf.boolean_mask(y_pred, mask))
        # Calculate contrastive loss (loss for labeled data)
        cons_loss = contrastive_loss(y_true, y_pred, num_labeled_per_class, 5.0)

        # Calculate alignment loss (loss for unlabeled data)
        ustart = num_labeled_per_class * num_classes
        num = []
        for u in range(num_unlabeled):
            num_u = []
            sstart = 0
            for j in range(num_classes):
                num_j = 0
                for i in range(num_labeled_per_class):
                    out1 = y_pred[ustart + u]
                    out2 = y_pred[sstart + i]
                    dot = tfm.multiply(out1, out2)
                    dot_prod = tfm.reduce_sum(dot)
                    #dot_prod = tfm.minimum(dot_prod, tf.constant(80.0))
                    num_j += tfm.exp(dot_prod)
                num_u.append(num_j)
                sstart += num_labeled_per_class
            denominator = tfm.reduce_sum(num_u)
            pij = tfm.divide_no_nan(num_u, denominator)
            pijlog = tfm.log(pij)
            h_vector = tfm.multiply(pij, pijlog)
            num.append(tfm.reduce_sum(h_vector))

        alignment_loss = tf.convert_to_tensor(num)
        alignment_loss = tfm.multiply(alignment_loss, -1.0)
        weighted_loss = tfm.reduce_mean(alignment_loss) + tfm.reduce_mean(supervised_loss) + cons_loss
        return weighted_loss

    return maskedloss_function

reconstructed_model = tf.keras.models.load_model("results/" + directory + "/model.h5",
        
                                                  custom_objects={'maskedloss_function': customLoss(num_labeled_per_class, num_classes, num_unlabeled)})


prediction = reconstructed_model.predict(x_test_source)[0]
prediction_labels = np.argmax(prediction, axis=1)
actual_labels = np.argmax(y_test_c_source, axis=1)
match_counter = np.sum(prediction_labels == actual_labels)

# Create a string containing the output for the source domain
source_output = (
    "Prediction for Source Domain\n"
    "============================\n"
    f"Match: {match_counter}\n"
    f"Total: {len(x_test_source)}\n"
    f"Accuracy: {(match_counter / len(x_test_source)) * 100}\n"
    f"F-1 score: {calculate_f1_score(prediction, y_test_c_source)}\n"
)

prediction = reconstructed_model.predict(x_test_target)[0]
prediction_labels = np.argmax(prediction, axis=1)
actual_labels = np.argmax(y_test_c_target, axis=1)
match_counter = np.sum(prediction_labels == actual_labels)

# Create a string containing the output for the target domain
target_output = (
    "Prediction for Target Domain\n"
    "============================\n"
    f"Match: {match_counter}\n"
    f"Total: {len(x_test_target)}\n"
    f"Accuracy: {(match_counter / len(x_test_target)) * 100}\n"
    f"F-1 score: {calculate_f1_score(prediction, y_test_c_target)}\n"
)

# Concatenate both outputs
full_output = source_output + "\n\n\n" + target_output

# Print the full output to the console
print(full_output)

# Save the full output to the file
output_file = "mda_results/" + directory + "/results.txt"
save_to_file(output_file, full_output)

# total time elapsed
total_time = time.time() - start_time
print("Total Time:", total_time, "seconds") 
