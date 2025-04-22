import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\rio.jpg')
img_gris = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\rio.jpg', cv.IMREAD_GRAYSCALE)

# Calcular el histograma
plt.figure(figsize=(8, 4))
plt.hist(img_gris.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histograma de Imagen en Escala de Grises')
plt.xlabel('Valor de Intensidad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Crear la imagen binaria: todo lo mayor a 19 en negro (0) y el resto en blanco (255)
img_bin = np.where(img_gris > 19, 0, 255).astype(np.uint8)
plt.imshow(img_bin, cmap='gray')
plt.title('Imagen Binaria')
plt.show()

# Convertimos la imagen binaria en una máscara para la parte blanca (el río)
mask_white = img_bin == 255  # Esta es la región que queremos cambiar a amarillo

# Ahora, en la imagen original, donde la máscara es blanca, cambiamos esos píxeles a amarillo
img[mask_white] = [0, 255, 255]  # Amarillo en BGR (OpenCV usa BGR)

# Mostrar la imagen resultante
cv.imshow("Resultado con Río Amarillo", img)
cv.waitKey(0)
cv.destroyAllWindows()

