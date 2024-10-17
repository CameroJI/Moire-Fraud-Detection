from keras.models import Model # type: ignore
from tensorflow.keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, Dense, Concatenate, Flatten, MaxPooling2D, Dropout # type: ignore

def create_model(height, width, depth):
    input_LL = Input(shape=(height, width, depth), name='LL_Input')
    input_HL = Input(shape=(height, width, depth), name='HL_Input')
    input_LH = Input(shape=(height, width, depth), name='LH_Input')
    input_HH = Input(shape=(height, width, depth), name='HH_Input')
    
    input_Scharr = Input(shape=(height, width, depth), name='Scharr_Input')
    input_Sobel = Input(shape=(height, width, depth), name='Sobel_Input')
    input_Gabor = Input(shape=(height, width, depth), name='Gabor_Input')

    input_R = Input(shape=(height, width, depth), name='R_Input')
    input_G = Input(shape=(height, width, depth), name='G_Input')
    input_B = Input(shape=(height, width, depth), name='B_Input')
    
    def conv_block(input_tensor):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        return Flatten()(x)
    
    x_LL = conv_block(input_LL)
    x_HL = conv_block(input_HL)
    x_LH = conv_block(input_LH)
    x_HH = conv_block(input_HH)

    x_Scharr = conv_block(input_Scharr)
    x_Sobel = conv_block(input_Sobel)
    x_Gabor = conv_block(input_Gabor)

    input_RGB = Concatenate(axis=-1)([input_R, input_G, input_B])
    x_RGB = conv_block(input_RGB)

    concatenated = Concatenate()([
        x_LL, x_HL, x_LH, x_HH, x_Scharr, x_Sobel, x_Gabor, x_RGB
    ])

    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[input_LL, input_HL, input_LH, input_HH, input_Scharr, input_Sobel, input_Gabor, input_R, input_G, input_B], outputs=predictions)