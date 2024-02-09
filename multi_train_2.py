import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm 
import os
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)




def load_images_from_folder(folder, label):
    data = []
    labels = []

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

def get_source_data(source_domain):
    source_folder = f'dataset/{source_domain}/train'
    fake_data, fake_labels = load_images_from_folder(os.path.join(source_folder, 'fake'), label=0)
    real_data, real_labels = load_images_from_folder(os.path.join(source_folder, 'real'), label=1)

    data = np.concatenate([fake_data, real_data], axis=0)
    labels = np.concatenate([fake_labels, real_labels], axis=0)

    labels_one_hot = to_categorical(labels, num_classes=2)

    return data, labels_one_hot

def get_target_data(target_domain):
    target_folder = f'dataset/{target_domain}/train'

    labeled_data_fake, labeled_labels_fake = load_images_from_folder(os.path.join(target_folder, 'labeled/fake'), label=0)
    labeled_data_real, labeled_labels_real = load_images_from_folder(os.path.join(target_folder, 'labeled/real'), label=1)
    unlabeled_data, dummy_labels = load_images_from_folder(os.path.join(target_folder, 'unlabeled'), label=-1)

    labeled_data = np.concatenate([labeled_data_fake, labeled_data_real], axis=0)
    labeled_labels = np.concatenate([labeled_labels_fake, labeled_labels_real], axis=0)

    #data = np.concatenate([labeled_data, unlabeled_data], axis=0)
    #labels = np.concatenate([labeled_labels, dummy_labels], axis=0)

    labels_one_hot = to_categorical(labeled_labels, num_classes=2)
    
    return labeled_data, labels_one_hot, unlabeled_data

def get_next_batch(data, labels, batch_size, labeled=True, shuffle=True):
    num_samples = len(data)
    
    if shuffle:
        indices = np.random.permutation(num_samples)
    else:
        indices = np.arange(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        batch_data = data[batch_indices]

        if labeled:
            batch_labels = labels[batch_indices]
            yield batch_data, batch_labels
        else:
            yield batch_data

def get_next_target_batch(labeled_data, labeled_labels, unlabeled_data, batch_size):
    num_labeled_samples = len(labeled_data)
    num_unlabeled_samples = len(unlabeled_data)

    # Shuffle labeled data and labels together
    indices_labeled = np.arange(num_labeled_samples)
    np.random.shuffle(indices_labeled)
    labeled_data = labeled_data[indices_labeled]
    labeled_labels = labeled_labels[indices_labeled]

    # Shuffle unlabeled data
    indices_unlabeled = np.arange(num_unlabeled_samples)
    np.random.shuffle(indices_unlabeled)
    unlabeled_data = unlabeled_data[indices_unlabeled]

    for start in range(0, num_unlabeled_samples, batch_size - 10):
        end = min(start + batch_size - 10, num_unlabeled_samples)
        batch_unlabeled_data = unlabeled_data[start:end]

        # Add 10 labeled samples to each batch
        batch_labeled_data = labeled_data[:10]
        batch_labeled_labels = labeled_labels[:10]

        labeled_data = labeled_data[10:]
        labeled_labels = labeled_labels[10:]

        # Concatenate labeled and unlabeled data
        batch_data = np.concatenate([batch_labeled_data, batch_unlabeled_data], axis=0)
        batch_labels = np.concatenate([batch_labeled_labels, np.zeros((batch_size - 10, 2))], axis=0)

        yield batch_data, batch_labels

def create_shared_model(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    return models.Sequential([
        base_model, 
        layers.Flatten(),
        layers.Dense(3072, activation='relu'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Activation('relu')
    ])

def create_domain_specific_classifier(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def gaussian_kernel(x, y, sigma):
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)

    exponent = -tf.reduce_sum(tf.square(x - y), axis=-1) / (2 * sigma**2)
    kernel_matrix = tf.exp(exponent)

    return kernel_matrix

def mmd_loss(source_features, target_features, sigma=1.0):
    source_kernel = gaussian_kernel(source_features, source_features, sigma)
    target_kernel = gaussian_kernel(target_features, target_features, sigma)
    cross_kernel = gaussian_kernel(source_features, target_features, sigma)

    mmd_loss = tf.reduce_mean(source_kernel) + tf.reduce_mean(target_kernel) - 2 * tf.reduce_mean(cross_kernel)
    mmd_loss = tf.maximum(mmd_loss, 0.0)  # Ensure non-negativity

    return mmd_loss

def custom_loss(y_true, y_pred):
    # Assuming the labels are one-hot encoded
    labeled_true = y_true[:, :2]  # Extract the labeled part of the target labels
    labeled_pred = y_pred[:, :2]  # Extract the predicted labeled part

    # Categorical crossentropy loss for the labeled part
    labeled_loss = tf.keras.losses.categorical_crossentropy(labeled_true, labeled_pred)

    return labeled_loss

def create_multi_source_model(input_shape, num_classes, num_source_domains):
    shared_model = create_shared_model(input_shape)

    # Create domain-specific classifiers for each source domain
    domain_specific_classifiers = [create_domain_specific_classifier(input_shape, num_classes) for _ in range(num_source_domains)]

    # Create domain inputs
    source_domain_inputs = [layers.Input(shape=input_shape) for _ in range(num_source_domains)]
    target_domain_input = layers.Input(shape=input_shape)

    # Shared feature extraction
    source_features = [shared_model(input_) for input_ in source_domain_inputs]
    target_features = shared_model(target_domain_input)

    # Classification heads for each domain
    source_domain_outputs = [classifier(feature) for feature, classifier in zip(source_features, domain_specific_classifiers)]
    
    # Classifier for the target domain
    target_classifier = create_domain_specific_classifier(input_shape, num_classes)
    target_domain_output = target_classifier(target_features)

    # Create the model
    model = models.Model(
        inputs=[*source_domain_inputs, target_domain_input],
        outputs=[*source_domain_outputs, target_domain_output]
    )

    return model

if __name__ == "__main__":

    # Load Training Data
    source_data_1, source_labels_1 = get_source_data('DeepFake')
    source_data_2, source_labels_2 = get_source_data('Face2Face')
    target_labeled_data, target_labels, target_unlabeled_data = get_target_data('FaceSwap')

    print("Data Loaded!!")

    # Preprocess data

    # Create the model
    input_shape = (128, 128, 3)  # Adjust according to your image size
    num_classes = 2  # Assuming 2 classes: 'fake' and 'real'
    num_source_domains = 2  # Assuming two source domains
    model = create_multi_source_model(input_shape, num_classes, num_source_domains)

    print("Model created!!")

    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss=['categorical_crossentropy', 'categorical_crossentropy', custom_loss, mmd_loss],
        metrics=[]
    )

    print("Model compiled!!")

    model.summary()

    # Train the model
    num_epochs = 50 
    batch_size = 100 

    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0 

        source1_gen = get_next_batch(source_data_1, source_labels_1, batch_size)
        source2_gen = get_next_batch(source_data_2, source_labels_2, batch_size)
        target_gen = get_next_target_batch(target_labeled_data, target_labels, target_unlabeled_data, batch_size)

        # Assuming you have a generator function for your data, adjust accordingly
        for _ in range(len(source_data_1) // batch_size):
            source_batch_1, labels_1 = next(source1_gen)
            source_batch_2, labels_2 = next(source2_gen)
            target_batch, target_labels_batch = next(target_gen)

            # Train on source domain batches
            loss = model.train_on_batch(
                [source_batch_1, source_batch_2, target_batch],
                [labels_1, labels_2, target_labels_batch]
            )
       
            total_loss += loss[0]
            num_batches += 1

        # Calculate and print the average loss for the epoch
        average_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')

    print("Model trained!!")
    end_time = time.time()  # Record the end time
    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds.")

    # Save the trained model
    model.save('multi_model.h5')
    print("Model saved!!")