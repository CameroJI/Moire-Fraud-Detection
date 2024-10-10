import os
import sys
import warnings
import argparse
import numpy as np
from tensorflow.keras.utils import img_to_array # type: ignore

from utils import preprocess_img, get_model, load_img

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):    
    modelPath = args.modelPath
    imgPath = args.imgPath
    
    height = args.height    
    width = args.width
    
    model = get_model(True, modelPath)
    img = load_img(imgPath, height=height, width=width)
    
    img = preprocess_img(img)
    prediction = predict_img(model, img)
    
    prediction_label = 'Real' if prediction == 0 else 'Ataque'
    print(prediction_label)

def predict_img(model, img, ResNet=False):
    if not ResNet:
        prediction = model.predict({
            'LL_Input': np.expand_dims(img['LL_Input'], axis=0),
            'LH_Input': np.expand_dims(img['LH_Input'], axis=0),
            'HL_Input': np.expand_dims(img['HL_Input'], axis=0),
            'HH_Input': np.expand_dims(img['HH_Input'], axis=0),
            'Scharr_Input': np.expand_dims(img['Scharr_Input'], axis=0),
            'Sobel_Input': np.expand_dims(img['Sobel_Input'], axis=0),
            'Gabor_Input': np.expand_dims(img['Gabor_Input'], axis=0),
            'RGB_Input': np.expand_dims(img['RGB_Input'], axis=0)
        }, verbose=0)

    else:
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array / 255.0
        prediction = model.predict(img_array, verbose=0)
        
    prediction_bin = (prediction[0] > 0.5).astype(int)[0]

    return prediction_bin, prediction[0][0]

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--modelPath', type=str, required=True, help='Directory for model Checkpoint')
    parser.add_argument('--imgPath', type=str, required=True, help='Directory for model Checkpoint')


    parser.add_argument('--height', type=int, help='Image height resize', default=800)
    parser.add_argument('--width', type=int, help='Image width resize', default=1400)
        
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))