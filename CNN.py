from keras.models import Model # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from keras.layers import Input, Conv2D, Dense, Concatenate, Flatten, MaxPooling2D, Dropout # type: ignore

def create_model(height, width):
    input_LL = Input(shape=(height, width, 1), name='LL_Input')
    input_HL = Input(shape=(height, width, 1), name='HL_Input')
    input_LH = Input(shape=(height, width, 1), name='LH_Input')
    input_HH = Input(shape=(height, width, 1), name='HH_Input')
    
    input_Scharr = Input(shape=(height, width, 1), name='Scharr_Input')
    input_Sobel = Input(shape=(height, width, 1), name='Sobel_Input')
    input_Gabor = Input(shape=(height, width, 1), name='Gabor_Input')

    input_RGB = Input(shape=(height, width, 3), name='RGB_Input')

    def conv_block(input_tensor):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        return Flatten()(x)

    # Convertir entradas de 1 canal a 3 canales usando convoluciones 1x1
    input_LL_rgb = Conv2D(3, (1, 1), padding='same')(input_LL)
    input_HL_rgb = Conv2D(3, (1, 1), padding='same')(input_HL)
    input_LH_rgb = Conv2D(3, (1, 1), padding='same')(input_LH)
    input_HH_rgb = Conv2D(3, (1, 1), padding='same')(input_HH)
    
    input_Scharr_rgb = Conv2D(3, (1, 1), padding='same')(input_Scharr)
    input_Sobel_rgb = Conv2D(3, (1, 1), padding='same')(input_Sobel)
    input_Gabor_rgb = Conv2D(3, (1, 1), padding='same')(input_Gabor)

    # Aplicar el bloque de convolución a cada entrada
    x_LL = conv_block(input_LL_rgb)
    x_HL = conv_block(input_HL_rgb)
    x_LH = conv_block(input_LH_rgb)
    x_HH = conv_block(input_HH_rgb)
    
    x_Scharr = conv_block(input_Scharr_rgb)
    x_Sobel = conv_block(input_Sobel_rgb)
    x_Gabor = conv_block(input_Gabor_rgb)

    x_RGB = conv_block(input_RGB)

    # Concatenar todas las salidas
    concatenated = Concatenate()([x_LL, x_HL, x_LH, x_HH, x_Scharr, x_Sobel, x_Gabor, x_RGB])

    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[input_LL, input_HL, input_LH, input_HH, input_Scharr, input_Sobel, input_Gabor, input_RGB], outputs=predictions)

def model_renNet(height, width, depth):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, depth))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # for layer in base_model.layers[-10:]:
    #     layer.trainable = True

    return Model(inputs=base_model.input, outputs=predictions)