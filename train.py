import sys
import argparse
import tensorflow as tf
from os.path import exists
from os import makedirs, walk
from keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

from utils import preprocess_img
from CNN import create_model_elements
from modelCallbacks import BatchCheckpointCallback, CustomImageDataGenerator


HEIGHT = 800
WIDTH = 1400

def main(args):
    global optimizer, WIDTH, HEIGHT
    
    datasetPath = args.datasetPath
    
    numEpochs = args.epochs
    save_iter = args.save_iter
    
    batch_size = args.batch_size
    
    checkpointPath = args.checkpointPath
    loadCheckPoint = args.loadCheckPoint
    
    HEIGHT = args.height
    WIDTH = args.width
    image_size = (HEIGHT, WIDTH)
        
    initial_learning_rate = args.learning_rate
    final_learning_rate = 1e-5
    decay_steps = count_img(datasetPath) // batch_size
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / decay_steps)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    if not exists(checkpointPath):
        makedirs(checkpointPath)
            
    checkpointPathModel = f"{checkpointPath}/model.keras"
    
    model = get_model(loadCheckPoint, checkpointPathModel)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'recall'])
    
    batchCheckpointCallback = BatchCheckpointCallback(batchesNumber=save_iter, path=checkpointPathModel)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    checkpoint = ModelCheckpoint(f'{checkpointPath}/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

    X_train = CustomImageDataGenerator(
        directory=datasetPath,
        batch_size=batch_size,
        image_size=(HEIGHT, WIDTH),
        preprocess_function=preprocess_img,
        class_mode='binary',
        classes={'Reales': 0, 'Ataque': 1}
    )
        
    # callbacks=[early_stopping, reduce_lr, checkpoint])
    
    model.fit(
        X_train, 
        epochs=numEpochs,
        callbacks=[batchCheckpointCallback], 
        )

def count_img(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')
    total_images = 0
    
    for root, dirs, files in walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                total_images += 1
    
    return total_images

def get_model(loadFlag, path):
    if loadFlag:
        model = load_model(path)
    else:
        model = create_model_elements(height=int(HEIGHT/8), width=int(WIDTH/8), depth=1)
        
    return model

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--datasetPath', type=str, help='Directory with dataset images folders.')
    
    parser.add_argument('--checkpointPath', type=str, help='Directory for model Checkpoint', default='./checkpoint/')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=10)
    parser.add_argument('--save_iter', type=int, help='Number of iterations to save the model', default=0)

    parser.add_argument('--batch_size', type=int, help='Batch size for epoch in training', default=32)
    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
    
    parser.add_argument('--learning_rate', type=float, help='Model learning rate for iteration', default=1e-3)
    
    parser.add_argument('--loadCheckPoint', action='store_true', default=False, help='load Checkpoint Model')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))