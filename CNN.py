from keras.models import Model # type: ignore
from tensorflow.keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, Dense, Concatenate, Flatten, MaxPooling2D, Dropout, Multiply, Average, BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.regularizers import l2

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
    x = Dropout(0.5)(x)
    x = Dense(64, ctivation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[input_LL, input_HL, input_LH, input_HH, input_Scharr, input_Sobel, input_Gabor, input_R, input_G, input_B], outputs=predictions)

def conv_block(input_tensor, filters, dropout_rate=0.3):
    x1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(filters, (5, 5), activation='relu', padding='same')(input_tensor)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(filters, (7, 7), activation='relu', padding='same')(input_tensor)
    x3 = BatchNormalization()(x3)
    x = Concatenate()([x1, x2, x3])
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    return x


def attention_block(x):
    attention = Conv2D(x.shape[-1], (1, 1), activation='relu', padding='same')(x)
    attention = BatchNormalization()(attention)
    attention = GlobalAveragePooling2D()(attention)
    attention = Dense(x.shape[-1], activation='sigmoid')(attention)
    attention = Multiply()([x, attention])
    
    return attention

def conv_block(input_tensor, filters, dropout_rate=0.3):
    x1 = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(filters, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(filters, (7, 7), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
    x3 = BatchNormalization()(x3)
    x = Concatenate()([x1, x2, x3])
    x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    return x

def attention_block(x):
    attention = Conv2D(x.shape[-1], (1, 1), activation='relu', padding='same')(x)
    attention = BatchNormalization()(attention)
    attention = GlobalAveragePooling2D()(attention)
    attention = Dense(x.shape[-1], activation='sigmoid')(attention)
    attention = Multiply()([x, attention])
    
    return attention

def create_new_model(height, width, depth):
    input_LL = Input(shape=(height, width, depth), name='LL_Input')
    input_HL = Input(shape=(height, width, depth), name='HL_Input')
    input_LH = Input(shape=(height, width, depth), name='LH_Input')
    input_HH = Input(shape=(height, width, depth), name='HH_Input')

    input_Scharr = Input(shape=(height, width, depth), name='Scharr_Input')
    input_Sobel = Input(shape=(height, width, depth), name='Sobel_Input')
    input_Gabor = Input(shape=(height, width, depth), name='Gabor_Input')

    input_R = Input(shape=(height, width, 1), name='R_Input')
    input_G = Input(shape=(height, width, 1), name='G_Input')
    input_B = Input(shape=(height, width, 1), name='B_Input')

    input_RGB = Concatenate(axis=-1)([input_R, input_G, input_B])
    
    branches = []

    for input_tensor in [input_LL, input_HL, input_LH, input_HH, input_Scharr, input_Sobel, input_Gabor]:
        x = conv_block(input_tensor, 32)
        x = attention_block(x)
        branches.append(Flatten()(x))
    
    x_RGB = conv_block(input_RGB, 32)
    x_RGB = attention_block(x_RGB)
    branches.append(Flatten()(x_RGB))

    avg_output = Average()(branches)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(avg_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[input_LL, input_HL, input_LH, input_HH, input_Scharr, input_Sobel, input_Gabor, input_R, input_G, input_B], outputs=predictions)