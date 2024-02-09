import numpy as np
import cv2
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm 
from sklearn.metrics import classification_report

from multi_train_2 import mmd_loss, get_target_data, get_source_data, create_multi_source_model

def load_images_from_folder(folder, label):
    data = []
    labels = []

    # Use tqdm to create a progress bar
    for filename in tqdm(os.listdir(folder), desc=f"Loading {folder}", unit="image"):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            if not filename.startswith('.'):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))  # Adjust the size according to your model input size
                img = img.astype('float32') / 255.0
                data.append(img)
                labels.append(label)

    return np.array(data), np.array(labels)

def get_test_data(source_domain):
    source_folder = f'dataset/{source_domain}/test'
    fake_data, fake_labels = load_images_from_folder(os.path.join(source_folder, 'fake'), label=0)
    real_data, real_labels = load_images_from_folder(os.path.join(source_folder, 'real'), label=1)

    data = np.concatenate([fake_data, real_data], axis=0)
    labels = np.concatenate([fake_labels, real_labels], axis=0)

    # Convert labels to one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=2)

    return data, labels_one_hot

def custom_loss(y_true, y_pred):
    # Assuming the labels are one-hot encoded
    labeled_true = y_true[:, :2]  # Extract the labeled part of the target labels
    labeled_pred = y_pred[:, :2]  # Extract the predicted labeled part

    # Categorical crossentropy loss for the labeled part
    labeled_loss = tf.keras.losses.categorical_crossentropy(labeled_true, labeled_pred)

    return labeled_loss

# Load the trained model
loaded_model = models.load_model('multi_model.h5', custom_objects={'custom_loss': custom_loss, 'mmd_loss': mmd_loss})
print('Model Loaded!!')

domains = ['DeepFake', 'Face2Face', 'FaceSwap']

for d in domains:
    # Load test data
    test_target_data, test_target_labels = get_test_data(d)
    print('\n\nTest Data Loaded for domain: ', d)

    # Make predictions on the test data
    test_predictions = loaded_model.predict([test_target_data, test_target_data, test_target_data])



    print('Results for Classifier 1')
    # Convert predictions to binary labels (0 for fake, 1 for real)
    predicted_labels = np.argmax(test_predictions[0], axis=1)

    # Convert labels to one-hot encoding
    predicted_labels_one_hot = to_categorical(predicted_labels, num_classes=2)

    # Calculate accuracy
    accuracy = accuracy_score(test_target_labels, predicted_labels_one_hot)
    print('Test Accuracy:', accuracy)

    # Count of correct predictions for each class
    correct_fake = np.sum((test_target_labels[:, 0] == 1) & (predicted_labels == 0))
    correct_real = np.sum((test_target_labels[:, 1] == 1) & (predicted_labels == 1))

    print('Correct Predictions for Fake:', correct_fake)
    print('Correct Predictions for Real:', correct_real)




    print('Results for Classifier 2')
    # Convert predictions to binary labels (0 for fake, 1 for real)
    predicted_labels = np.argmax(test_predictions[1], axis=1)

    # Convert labels to one-hot encoding
    predicted_labels_one_hot = to_categorical(predicted_labels, num_classes=2)

    # Calculate accuracy
    accuracy = accuracy_score(test_target_labels, predicted_labels_one_hot)
    print('Test Accuracy:', accuracy)

    # Count of correct predictions for each class
    correct_fake = np.sum((test_target_labels[:, 0] == 1) & (predicted_labels == 0))
    correct_real = np.sum((test_target_labels[:, 1] == 1) & (predicted_labels == 1))

    print('Correct Predictions for Fake:', correct_fake)
    print('Correct Predictions for Real:', correct_real)




    print('Results for Classifier 3')
    # Convert predictions to binary labels (0 for fake, 1 for real)
    predicted_labels = np.argmax(test_predictions[2], axis=1)

    # Convert labels to one-hot encoding
    predicted_labels_one_hot = to_categorical(predicted_labels, num_classes=2)

    # Calculate accuracy
    accuracy = accuracy_score(test_target_labels, predicted_labels_one_hot)
    print('Test Accuracy:', accuracy)

    # Count of correct predictions for each class
    correct_fake = np.sum((test_target_labels[:, 0] == 1) & (predicted_labels == 0))
    correct_real = np.sum((test_target_labels[:, 1] == 1) & (predicted_labels == 1))

    print('Correct Predictions for Fake:', correct_fake)
    print('Correct Predictions for Real:', correct_real)

