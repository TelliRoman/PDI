import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
##2-A
'''
# Función de callback para manejar los clics
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # Si se presiona el botón izquierdo
        print(f"Coordenadas: ({x}, {y}) - Intensidad: {imagen[y, x]}")
        
        # Mostrar el punto en la imagen
        cv.circle(imagen, (x, y), 1, (0, 255, 255), -1)
        cv.imshow("Imagen", imagen)

imagen = cv.imread("Guia1\camino.tif", cv.IMREAD_GRAYSCALE)
# Mostrar la imagen en una ventana
cv.imshow("Imagen", imagen)

# Vincular la función de clic con la ventana
cv.setMouseCallback("Imagen", click_event)

cv.waitKey(0)  # Esperar a que se presione una tecla
cv.destroyAllWindows()'''

##2-B
imagen = cv.imread("Guia1\camino.tif", cv.IMREAD_GRAYSCALE)
line=100
plt.figure()
plt.imshow(imagen,cmap='gray')
plt.plot([0,imagen.shape[1]],[line,line]) #recibe ((xmin,xmax),(ymin,ymax))

plt.figure()
plt.plot(imagen[line,:])
plt.title("Perfil de intensidad")
plt.show()

# Definir los puntos del segmento (x1, y1) -> (x2, y2)
x1, y1 = 0, 0  # Punto inicial
x2, y2 = imagen.shape[1]-1,imagen.shape[0]-1  # Punto final (misma Y para un perfil horizontal)

# Obtener los valores de intensidad en la línea
num_puntos = max(abs(x2 - x1), abs(y2 - y1))  # Número de puntos a muestrear
x_values = np.linspace(x1, x2, num_puntos, dtype=int)
y_values = np.linspace(y1, y2, num_puntos, dtype=int)

# Lista para almacenar las intensidades
intensities = []

# Recorrer los puntos de la línea y obtener las intensidades
for i in range(len(x_values)):
    x = x_values[i]
    y = y_values[i]
    intensities.append(imagen[y, x])

# Graficar la imagen y el segmento
plt.figure()
plt.imshow(imagen, cmap="gray")
plt.plot([x1, x2], [y1, y2], color='red')  # Dibujar la línea
plt.title("Imagen con segmento de interés")

plt.figure()
plt.plot(intensities, color='blue')
plt.title("Perfil de Intensidad")

plt.show()

