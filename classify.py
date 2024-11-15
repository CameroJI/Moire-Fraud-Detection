import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

folders = ['fail', 'pass']
root = '/Users/jorgecamero/Moire-Fraud-Detection'

df = pd.read_excel('/Users/jorgecamero/Desktop/Moire_302.xlsx', sheet_name=0)
df['label real'] = df['label real'].astype(str)

# Empezar desde el índice
start_index = 0
current_index = start_index

prediction = None

def on_key(event):
    global prediction, current_index
    if event.key == 'p':  # Tecla 'p' para PASS
        prediction = 'PASS'
        current_index = min(len(df) - 1, current_index + 1)  # Avanzar a la siguiente imagen
    elif event.key == 'a':  # Tecla 'a' para FAIL
        prediction = 'FAIL'
        current_index = min(len(df) - 1, current_index + 1)  # Avanzar a la siguiente imagen
    elif event.key == 'left':  # Flecha izquierda para retroceder
        current_index = max(start_index, current_index - 1)
    elif event.key == 'right':  # Flecha derecha para avanzar
        current_index = min(len(df) - 1, current_index + 1)

    # Cerrar la ventana y actualizar la imagen
    plt.close()
    plot_image()

def plot_image():
    global current_index, prediction
    file_name = df.at[current_index, 'file']
    
    for folder in folders:
        file_path = os.path.join(root, folder, file_name)
        
        if os.path.exists(file_path):
            # Cargar y mostrar la imagen
            image = Image.open(file_path)
            plt.imshow(image)
            plt.axis('off')
            plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            plt.show(block=False)  # Muestra la imagen sin bloquear

            # Si hay una predicción, se guarda en el archivo
            if prediction is not None:
                if df.at[current_index - 1, 'label real'] != prediction:
                    df.at[current_index - 1, 'label real'] = prediction
                    df.to_excel('/Users/jorgecamero/Desktop/Moire_302.xlsx', index=False)  # Guardar tras cada cambio
                    prediction = None  # Reiniciar la predicción

            break

def clear_console():
    """Limpia la consola (funciona en Windows y UNIX)."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Bucle principal para iterar sobre las filas comenzando en el start_index
while current_index < len(df):
    plot_image()

    # Limpiar la consola y mostrar el progreso
    clear_console()
    print(f"Procesando archivo {current_index + 1}/{len(df)}\t{df.at[current_index, 'prediction nueva']}")

    # Pausar brevemente para permitir la interacción con las teclas
    plt.pause(0.1)  # Pausar brevemente para que se muestren las teclas y la imagen

# Después de procesar todas las imágenes, guardar el archivo
df.to_excel('/Users/jorgecamero/Desktop/Moire_303.xlsx', index=False)
print("Archivo Excel actualizado con las predicciones.")