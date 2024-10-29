import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from os.path import exists
from os import makedirs, walk
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

from utils import preprocess_img, preprocess_augmentation_img, get_model, detect_left_face, load_img
from modelCallbacks import BatchCheckpointCallback, CustomImageDataGenerator

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

HEIGHT = 800
WIDTH = 1400

def main(args):
    global optimizer, WIDTH, HEIGHT
    
    datasetPath = args.datasetPath
    
    monitor = args.monitor
    monitor_mode = args.monitor_mode
    
    numEpochs = args.epochs
    save_iter = args.save_iter
    
    batch_size = args.batch_size
    
    checkpointPath = args.checkpointPath
    loadCheckpoint = args.loadCheckpoint
    dataset_augmentation = args.dataset_augmentation
    left_face_only = args.left_face_only
    
    unfreeze_layers = args.unfreeze_layers
    
    HEIGHT = args.height
    WIDTH = args.width
    image_size = (HEIGHT, WIDTH)
        
    learning_rate = args.learning_rate
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    
    if not exists(checkpointPath):
        makedirs(checkpointPath)
            
    checkpointPathModel = f"{checkpointPath}/model.keras"
    
    model = get_model(loadCheckpoint, checkpointPathModel, unfreeze_layers, HEIGHT, WIDTH)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'recall'])
    
    batchCheckpointCallback = BatchCheckpointCallback(batchesNumber=save_iter, path=checkpointPathModel)
    
    early_stopping = EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True, mode=monitor_mode)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.05, patience=2, mode=monitor_mode)
    checkpoint = ModelCheckpoint(f'{checkpointPath}/best_model.keras', monitor=monitor, save_best_only=True, verbose=1, mode=monitor_mode)
    
    images, labels = load_data(datasetPath, left_face_only)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, shuffle=True)

    train_generator, val_generator = get_generator(X_train, y_train, X_val, y_val, batch_size, image_size, dataset_augmentation)
    
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=numEpochs,
        callbacks=[batchCheckpointCallback, early_stopping, reduce_lr, checkpoint]
    )
    
def load_data(datasetPath, left_face_only):
    images = []
    labels = []
    folders = [folder for folder in os.listdir(datasetPath) if os.path.isdir(os.path.join(datasetPath, folder))]
    for folder in tqdm(folders, desc="Procesando carpetas"):
        folder_path = os.path.join(datasetPath, folder)
        img_list = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for img_file in tqdm(img_list, desc=f"Procesando imágenes en {folder}", leave=False):
            img_path = os.path.join(folder_path, img_file)
            images.append(img_path)
            
            if left_face_only:
                img = load_img(img_path, width=WIDTH, height=HEIGHT)
                img_crop = detect_left_face(img=img)                    
                labels.append(0 if img_crop is not None and folder == "Reales" else 1)
            else:
                labels.append(0 if folder == 'Reales' else 1)
    return np.array(images), np.array(labels)

def get_generator(X_train, y_train, X_val, y_val, batch_size, image_size, dataset_augmentation=False):
    if dataset_augmentation:
                    preprocess_function = lambda image: preprocess_augmentation_img(image, height=HEIGHT, width=WIDTH)
    else:
        preprocess_function = lambda image: preprocess_img(image, height=HEIGHT, width=WIDTH)
        
    valid_preprocess_function = lambda image: preprocess_img(image, height=HEIGHT, width=WIDTH)
        
    train_generator = CustomImageDataGenerator(
        image_paths=X_train,
        labels=y_train,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_function=preprocess_function,
        class_mode='binary'
        )

    val_generator = CustomImageDataGenerator(
        image_paths=X_val,
        labels=y_val,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_function=valid_preprocess_function,
        class_mode='binary'
        )
    
    return train_generator, val_generator

def count_img(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')
    total_images = 0
    
    for root, dirs, files in walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                total_images += 1
    
    return total_images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datasetPath', type=str, help='Directory with dataset images folders.')
    
    parser.add_argument('--monitor', type=str, help='Monitor for learning model', default='val_loss')
    parser.add_argument('--monitor_mode', type=str, help='Monitor mode for learning model', default='min')
    parser.add_argument('--checkpointPath', type=str, help='Directory for model Checkpoint', default='./checkpoint/')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--save_iter', type=int, help='Number of iterations to save the model', default=1000)

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    
    parser.add_argument('--unfreeze_layers', type=int, help='Number of layers to unfreeze in the model', default=0)
    
    parser.add_argument('--learning_rate', type=float, help='Model learning rate for iteration', default=1e-3)
    
    parser.add_argument('--dataset_augmentation', action='store_true', default=False, help='Use data augmentation on dataset')
    parser.add_argument('--left_face_only', action='store_true', default=False, help='Use data augmentation on dataset')
    parser.add_argument('--loadCheckpoint', action='store_true', default=False, help='load Checkpoint Model')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))