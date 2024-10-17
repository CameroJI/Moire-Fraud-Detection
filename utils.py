import cv2
import pywt
import numpy as np
import tensorflow as tf
import mediapipe as mp
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

from CNN import create_model, create_new_model

mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
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

    return image[
        offset_height : offset_height + target_height,
        offset_width : offset_width + target_width,
    ]

def wavelet_function(img):
    coeffs2 = pywt.wavedec2(img, 'bior2.2', level=3)
    LL, (HL, LH, HH) = coeffs2[0], coeffs2[1]
    return LL, LH, HL, HH

def resize(component, target_height, target_width):
    return tf.image.resize(
        component, (int(target_height), int(target_width)), method='bilinear'
    )

def preprocess_augmentation_img(image, height=200, width=350):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.65, upper=1.35)
        
    r_channel = image[..., 0]
    g_channel = image[..., 1]
    b_channel = image[..., 2]
    
    image = tf.image.rgb_to_grayscale(image)
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
    
    LL_resized = resize(LL_tensor, height, width)
    LH_resized = resize(LH_tensor, height, width)
    HL_resized = resize(HL_tensor, height, width)
    HH_resized = resize(HH_tensor, height, width)
    imgScharr_resized = resize(imgScharr_tensor, height, width)
    imgSobel_resized= resize(imgSobel_tensor, height, width)
    imgGabor_resized = resize(imgGabor_tensor, height, width)

    r_resized = resize(r_tensor, height, width)
    g_resized = resize(g_tensor, height, width)
    b_resized = resize(b_tensor, height, width)
        
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
    
def preprocess_img(image, height=200, width=350):
    img_crop = detect_left_face(image)
    image = tf.convert_to_tensor(image)  
    if img_crop is not None:
        r_channel = img_crop[..., 0]
        g_channel = img_crop[..., 1]
        b_channel = img_crop[..., 2]
    else:
        r_channel = image[..., 0]
        g_channel = image[..., 1]
        b_channel = image[..., 2]
   
    image = tf.image.rgb_to_grayscale(image)
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
    
    LL_resized = resize(LL_tensor, height, width)
    LH_resized = resize(LH_tensor, height, width)
    HL_resized = resize(HL_tensor, height, width)
    HH_resized = resize(HH_tensor, height, width)
    imgScharr_resized = resize(imgScharr_tensor, height, width)
    imgSobel_resized = resize(imgSobel_tensor, height, width)
    imgGabor_resized = resize(imgGabor_tensor, height, width)
    
    r_resized = resize(r_tensor, height, width)
    g_resized = resize(g_tensor, height, width)
    b_resized = resize(b_tensor, height, width)

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
    
def get_model(loadFlag, path, unfreeze_layers=0, height=800, width=1400):
    if not loadFlag:
        return (
            create_new_model(height=int(height), width=int(width), depth=1))
    model = load_model(path)

    if unfreeze_layers > 0:
        for i, layer in enumerate(model.layers):
            layer.trainable = i >= len(model.layers) - unfreeze_layers

    return model

def load_img(path, height=800, width=1400):
    return image.load_img(path, target_size=(height, width))

def detect_left_face(img):
    img_array = np.array(image.img_to_array(img)).astype(np.uint8)
    results = detector.process(img_array)

    if results.detections is None:
        return None
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = img_array.shape
        x, y, width, height = (bboxC.xmin * w, bboxC.ymin * h, bboxC.width * w, bboxC.height * h)
        h_prop, y_prop = (int(h/5), int(y/2))
        if x < h/2:
            img_crop = img_array[int(y)-y_prop:int(y + height)+y_prop, int(x)-h_prop:int(x + width)+h_prop]
            
    return img_crop