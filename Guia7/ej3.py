import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(100000)
'''clicked = False
fila = None
columna = None
# Función callback para capturar eventos del mouse
def mostrar_posicion(event, x, y, flags, param):
    global fila, columna, clicked
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(f"Mouse en fila={y}, columna={x}")
        print(f"Valor de intensidad: {img[y, x]}")
        fila = y
        columna = x
        clicked = True'''

def crecimiento_regiones(img, mask, fila, columna, semilla_val, rangogris=10):
    # Control de límites
    if fila < 0 or fila >= img.shape[0] or columna < 0 or columna >= img.shape[1]:
        return
    # Si ya está marcado, salir
    if mask[fila, columna] == 255:
        return
    # Si no cumple el criterio de similitud, salir
    if abs(int(img[fila, columna]) - int(semilla_val)) > rangogris:
        return
    # Marca el píxel
    mask[fila, columna] = 255
    # Llama recursivamente a los vecinos (8-conectividad)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0: #PARA NO LLAMAR CUANDO LOS 2 SON CERO
                crecimiento_regiones(img, mask, fila+dx, columna+dy, semilla_val, rangogris)
                # Aquí pasas el valor del píxel actual, no el de la semilla original
                # para comparar con el vecino y no con la semilla actual
                #crecimiento_regiones(img, mask, fila+dx, columna+dy, img[fila, columna], rangogris)
                #Si comparas siempre con la semilla original, la región es más estricta.
                #Si comparas con el último píxel incluido, la región puede adaptarse a cambios suaves y crecer más.

# Cargar imagen y máscara
#img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\rio.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\rmn.jpg', cv.IMREAD_GRAYSCALE)

#DOS OPCIONES PARA MOSTRAR LA IMAGEN Y OBTENER LA SEMILLA CON CLICK
'''# Mostrar imagen y asignar el callback
cv.namedWindow("Imagen")
cv.setMouseCallback("Imagen", mostrar_posicion)

while True:
    cv.imshow("Imagen", img)
    if clicked:  # Sale si hiciste doble click
        break
    if cv.waitKey(1) & 0xFF == 27:  # ESC para salir
        break
cv.destroyAllWindows()
if fila is None or columna is None:
    print("No seleccionaste una semilla.")
    exit()
semilla_val = img[fila, columna]'''

# Mostrar imagen y esperar click
plt.imshow(img, cmap='gray')
plt.title("Haz click para seleccionar la semilla")
punto = plt.ginput(1)  # Espera un click
plt.close()
# Obtener fila y columna (recuerda que plt.ginput da (x, y))
columna, fila = int(punto[0][0]), int(punto[0][1])
semilla_val = img[fila, columna]

# Rango de inclusión (puedes variar este valor)
mask = np.zeros(img.shape, dtype=np.uint8)
#rangogris = 9 #Buen rango para ver zona gris del cerebro
rangogris = 60 #Buen rango para ver zona blanca del cerebro

crecimiento_regiones(img, mask, fila, columna, semilla_val, rangogris)

# Visualización en pseudocolor
img_color = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img_color[mask == 255] = [255, 255, 0]
plt.imshow(img_color)
plt.show()

'''
Crecimiento de regiones:
Funcionamiento:
Se recorre recursivamente la imagen desde la semilla, marcando los píxeles conectados que cumplen el criterio.
Solo los píxeles adyacentes y similares en gris a la semilla se incluyen en la región.
El resultado es una máscara que marca la región crecida, visualizada en pseudocolor.
Ventaja:
Permite segmentar regiones específicas y conectadas, útil para extraer objetos de interés con bordes difusos o formas irregulares.

Umbralización global:
Funcionamiento:
Todos los píxeles con valor menor al umbral se colorean, el resto queda igual.
No se considera conectividad ni posición, solo el valor de gris.
El resultado es una imagen donde las zonas bajo el umbral se resaltan.
Ventaja:
Es rápido y simple para separar fondo/objeto cuando hay buen contraste global, pero no distingue regiones conectadas ni formas.

Característica	            Crecimiento de regiones  ||| 	Umbralización global 
Criterio de inclusión	Similitud local (rango de gris y conectividad) |||	Umbral global de gris
Segmenta regiones	    Sí, solo regiones conectadas  |||	No, afecta toda la imagen
Aplicación típica	Objetos de forma irregular, bordes difusos	||| Imágenes con contraste claro
Visualización	    Pseudocolor sobre región crecida |||	Color sobre píxeles bajo umbral
Resumen:
El crecimiento de regiones es más selectivo y útil para segmentar objetos conectados a partir de una semilla.
La umbralización global es más simple y rápida, pero menos precisa para regiones complejas o con ruido.
'''