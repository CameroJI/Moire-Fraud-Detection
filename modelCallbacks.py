import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import callbacks

class BatchCheckpointCallback(callbacks.Callback):
    def __init__(self, batchesNumber, path):
        super(BatchCheckpointCallback, self).__init__()
        self.count = 0
        self.batchesNumber = batchesNumber
        self.modelSaveBatchPath = path

    def on_batch_end(self, batch, logs=None):
        self.count += 1
        if self.count % self.batchesNumber == 0:
            print('\nGuardando modelo... ', end='')
            self.model.save(self.modelSaveBatchPath)
            print(f'Modelo guardado en {self.modelSaveBatchPath}')
        
class CustomImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, image_size, preprocess_function, class_mode='binary', shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.preprocess_function = preprocess_function
        self.class_mode = class_mode
        self.indexes = np.arange(len(self.image_paths))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_image_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = np.array([self.labels[i] for i in batch_indexes])

        batch_images_dict = {
            'LL_Input': [],
            'LH_Input': [],
            'HL_Input': [],
            'HH_Input': [],
            'Scharr_Input': [],
            'Sobel_Input': [],
            'Gabor_Input': []
        }
        
        for image_path in batch_image_paths:
            components = self._load_and_preprocess_image(image_path)
            for key in batch_images_dict:
                batch_images_dict[key].append(components[key])
        
        batch_images_dict = {key: np.stack(value) for key, value in batch_images_dict.items()}
        
        return batch_images_dict, batch_labels
    
    def _get_image_paths(self):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        image_paths = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    try:
                        image = tf.io.read_file(full_path)
                        tf.image.decode_image(image, channels=3)
                        image_paths.append(full_path)
                    except Exception as e:
                        print(f"Ignoring invalid image file: {full_path}, error: {e}")
                else:
                    print(f"Ignoring non-image file: {os.path.join(root, file)}")
        return image_paths

    def _load_and_preprocess_image(self, image_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            components = self.preprocess_function(image)
            
            required_keys = {'LL_Input', 'LH_Input', 'HL_Input', 'HH_Input', 'Scharr_Input', 'Sobel_Input', 'Gabor_Input'}
            if not all(key in components for key in required_keys):
                raise ValueError("Preprocessing function must return a dictionary with keys 'LL_Input', 'LH_Input', 'HL_Input', 'HH_Input', 'Scharr_Input', 'Sobel_Input' and 'Gabor_Input'.")
            
            return components
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
