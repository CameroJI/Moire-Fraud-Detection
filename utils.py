import cv2
import pywt
import numpy as np
import tensorflow as tf
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

from CNN import create_model, model_renNet

def scharr(img):
    image_np = img.numpy()

    scharr_x = cv2.Scharr(image_np, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image_np, cv2.CV_64F, 0, 1)

    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(scharr_combined)
    
    return scharr_combined

def sobel(img):
    image_np = img.numpy()
    
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)

    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined)
    
    return sobel_combined

def gabor(img, ksize=31, sigma=6.0, theta=0, lambd=4.0, gamma=0.2, psi=0.0):
    image_np = img.numpy()
        
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
    gabor_filtered = cv2.filter2D(image_np, cv2.CV_64F, gabor_kernel)
    gabor_filtered = np.uint8(np.abs(gabor_filtered))
    
    return gabor_filtered

def crop(image, target_height, target_width):
    image = tf.convert_to_tensor(image)
    
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    
    offset_height = (original_height - target_height) // 2
    offset_width = (original_width - target_width) // 2
    
    cropped_image = image[
        offset_height:offset_height + target_height,
        offset_width:offset_width + target_width,
    ]
    
    return cropped_image

def wavelet_function(img):
    coeffs2 = pywt.wavedec2(img, 'bior2.2', level=3)
    LL, (HL, LH, HH) = coeffs2[0], coeffs2[1]
    return LL, LH, HL, HH

def resize(component, target_height, target_width):
    component_resized = tf.image.resize(component, (int(target_height), int(target_width)), method='bilinear')
    return component_resized

def preprocess_augmentation_img(image, height=800, width=1400):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.65, upper=1.35)
    
    imageCrop = crop(image, height, width)
    
    r_channel = imageCrop[..., 0]
    g_channel = imageCrop[..., 1]
    b_channel = imageCrop[..., 2]
    
    image = tf.image.rgb_to_grayscale(imageCrop)
    imgScharr = scharr(image)
    imgSobel = sobel(image)
    imgGabor = gabor(image)
    image = tf.image.per_image_standardization(image)
    image = tf.squeeze(image, axis=-1)
    
    LL, LH, HL, HH = wavelet_function(image)
    
    LL_tensor = np.expand_dims(LL, axis=-1)
    LH_tensor = np.expand_dims(LH, axis=-1)
    HL_tensor = np.expand_dims(HL, axis=-1)
    HH_tensor = np.expand_dims(HH, axis=-1)
    imgScharr_tensor = np.expand_dims(imgScharr, axis=-1)
    imgSobel_tensor = np.expand_dims(imgSobel, axis=-1)
    imgGabor_tensor = np.expand_dims(imgGabor, axis=-1)
    r_tensor = np.expand_dims(r_channel, axis=-1)
    g_tensor = np.expand_dims(g_channel, axis=-1)
    b_tensor = np.expand_dims(b_channel, axis=-1)
    
    LL_resized = resize(LL_tensor, height/8, width/8)
    LH_resized = resize(LH_tensor, height/8, width/8)
    HL_resized = resize(HL_tensor, height/8, width/8)
    HH_resized = resize(HH_tensor, height/8, width/8)
    imgScharr_resized = resize(imgScharr_tensor, height/8, width/8)
    imgSobel_resized= resize(imgSobel_tensor, height/8, width/8)
    imgGabor_resized = resize(imgGabor_tensor, height/8, width/8)

    r_resized = resize(r_tensor, height/8, width/8)
    g_resized = resize(g_tensor, height/8, width/8)
    b_resized = resize(b_tensor, height/8, width/8)
        
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized,
        'Scharr_Input': imgScharr_resized,
        'Sobel_Input': imgSobel_resized,
        'Gabor_Input': imgGabor_resized,
        'R_Input': r_resized,
        'G_Input': g_resized,
        'B_Input': b_resized
    }
    
def preprocess_img(image, height=800, width=1400):
    imageCrop = crop(image, height, width)
    
    r_channel = imageCrop[..., 0]
    g_channel = imageCrop[..., 1]
    b_channel = imageCrop[..., 2]
    
    image = tf.image.rgb_to_grayscale(imageCrop)
    imgScharr = scharr(image)
    imgSobel = sobel(image)
    imgGabor = gabor(image)
    image = tf.image.per_image_standardization(image)
    image = tf.squeeze(image, axis=-1)
    
    LL, LH, HL, HH = wavelet_function(image)
    
    LL_tensor = np.expand_dims(LL, axis=-1)
    LH_tensor = np.expand_dims(LH, axis=-1)
    HL_tensor = np.expand_dims(HL, axis=-1)
    HH_tensor = np.expand_dims(HH, axis=-1)
    imgScharr_tensor = np.expand_dims(imgScharr, axis=-1)
    imgSobel_tensor = np.expand_dims(imgSobel, axis=-1)
    imgGabor_tensor = np.expand_dims(imgGabor, axis=-1)
    r_tensor = np.expand_dims(r_channel, axis=-1)
    g_tensor = np.expand_dims(g_channel, axis=-1)
    b_tensor = np.expand_dims(b_channel, axis=-1)
    
    LL_resized = resize(LL_tensor, height/8, width/8)
    LH_resized = resize(LH_tensor, height/8, width/8)
    HL_resized = resize(HL_tensor, height/8, width/8)
    HH_resized = resize(HH_tensor, height/8, width/8)
    imgScharr_resized = resize(imgScharr_tensor, height/8, width/8)
    imgSobel_resized= resize(imgSobel_tensor, height/8, width/8)
    imgGabor_resized = resize(imgGabor_tensor, height/8, width/8)

    r_resized = resize(r_tensor, height/8, width/8)
    g_resized = resize(g_tensor, height/8, width/8)
    b_resized = resize(b_tensor, height/8, width/8)
        
    return {
        'LL_Input': LL_resized,
        'LH_Input': LH_resized,
        'HL_Input': HL_resized,
        'HH_Input': HH_resized,
        'Scharr_Input': imgScharr_resized,
        'Sobel_Input': imgSobel_resized,
        'Gabor_Input': imgGabor_resized,
        'R_Input': r_resized,
        'G_Input': g_resized,
        'B_Input': b_resized
    }
    
def get_model(loadFlag, path, ResNet50=False, height=800, width=1400):
    if loadFlag:
        model = load_model(path)
    else:
        if not ResNet50:
            model = create_model(height=int(height/8), width=int(width/8), depth=1)
        else:
            model = model_renNet(height=height, width=width, depth=3)
    return model

def load_img(path, height=800, width=1400):
    img = image.load_img(path, target_size=(height, width))
    return img