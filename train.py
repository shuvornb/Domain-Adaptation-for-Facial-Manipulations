import numpy as np
import pandas as pd
from skimage import io
import cv2
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Activation, Lambda
import tensorflow.math as tfm
import tensorflow as tf
import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import f1_score
import keras
import time
import sys
import os

# Start the timer
start_time = time.time()

# Constants
img_rows, img_cols = 128, 128
num_classes = 2
num_labeled_per_class = 110
num_unlabeled = 30
batch_size = 250
input_shape = (img_rows, img_cols, 3)
epochs = 75
source = sys.argv[1]
target = sys.argv[2]
trial_no = sys.argv[3]

# Load processed data from disk
x_train = np.load("processed_data/x_train.npy")
y_train_c = np.load("processed_data/y_train_c.npy")
y_train_d = np.load("processed_data/y_train_d.npy")

print('All data loaded!')

# helper functions
def calculate_f1_score(prediction, actual):
	prediction_labels = np.argmax(prediction, axis=1)
	actual_labels = np.argmax(actual, axis=1)
	return f1_score(actual_labels, prediction_labels)

def flip_gradient(x, l=1.0):
    positive_path = tf.stop_gradient(x * (1 + l))
    negative_path = -x * l
    return positive_path + negative_path

def customLoss(num_labeled_per_class, num_classes, num_unlabeled):
    
    def maskedloss_function(y_true, y_pred):

        # Calculate supervised loss (loss for labeled data)
        mask = tf.concat([tf.ones(num_labeled_per_class * num_classes, dtype=tf.bool),
                           tf.zeros(num_unlabeled, dtype=tf.bool)], axis=0)
        supervised_loss = K.categorical_crossentropy(tf.boolean_mask(y_true, mask),
                                                      tf.boolean_mask(y_pred, mask))
         
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
                    dot_prod = tfm.minimum(dot_prod, tf.constant(80.0))
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
        weighted_loss = tfm.reduce_mean(alignment_loss) + tfm.reduce_mean(supervised_loss)
        
        #weighted_loss = tfm.reduce_mean(supervised_loss)
        return weighted_loss
        
        #return supervised_loss
    return maskedloss_function

# Xception base model
base_model = Xception(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')

# Train base model layers
base_model.trainable = True

# Category classifier
category_classifier = base_model.output
category_classifier = Flatten()(category_classifier)
#category_classifier = BatchNormalization()(category_classifier)
category_classifier = Dense(512, activation='relu', name='ccdense')(category_classifier)
category_classifier = Dense(num_classes, activation='softmax', name='cc')(category_classifier)

# Domain classifier
domain_classifier = Lambda(lambda x: flip_gradient(x))(base_model.output)
#domain_classifier = base_model.output
domain_classifier = Flatten()(domain_classifier)
#domain_classifier = BatchNormalization()(domain_classifier)
domain_classifier = Dense(512, activation='relu')(domain_classifier)
#domain_classifier = Lambda(lambda x: flip_gradient(x))(domain_classifier)  # Gradient reversal layer
domain_classifier = Dense(2, activation='softmax', name='dc')(domain_classifier)

# Source classification model
category_model = Model(inputs=base_model.inputs, outputs=category_classifier)
category_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
                                        loss=customLoss(num_labeled_per_class, num_classes, num_unlabeled),
                                        metrics=['accuracy'])

# Domain classification model
domain_model = Model(inputs=base_model.inputs, outputs=domain_classifier)
domain_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy'])

# Combined model
combined_outputs = [category_classifier, domain_classifier]
combined_model = Model(inputs=base_model.inputs, outputs=combined_outputs)
combined_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
                            loss={'cc': customLoss(num_labeled_per_class, num_classes, num_unlabeled),
                                  'dc': 'categorical_crossentropy'},
                            loss_weights={'cc': 10.0, 'dc': 1.0},
                            metrics=['accuracy'])

# Print model summary
print(combined_model.summary())

es = tf.keras.callbacks.EarlyStopping(
	monitor='loss',
    patience=10,
    mode='min',
    min_delta=0.001
)
'''
curr_time = datetime.datetime.now()
mc = tf.keras.callbacks.ModelCheckpoint(
	"saved_models/best_m_" + str(curr_time) + ".h5",
	monitor = 'loss',
	verbose = 1,
	save_best_only = True,
	mode = 'min'
)
'''
# Train the model
history = combined_model.fit(
	x_train,
    {'cc': y_train_c, 'dc': y_train_d},
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    callbacks=[es]
)

# Save the model
directory = source + "_" + target + "_" + str(trial_no)
parent_dir = "/home/mdshamimseraj/Desktop/FakeImageDetection/UDA/results/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
combined_model.save("results/" + directory + "/model.h5")

# total time elapsed
total_time = time.time() - start_time
print("Total Time:", total_time, "seconds")    
