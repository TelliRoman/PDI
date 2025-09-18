import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\cameraman.tif',cv.IMREAD_GRAYSCALE)

### OPCION 1 GENERAR EL FILTRO GAUSSIANO Y LUEGO APLICARLO ###
    # Crear kernel 1D gaussiano de tamaño 5, sigma=1
kernel = cv.getGaussianKernel(5, 3)

    # Convertir a kernel 2D
kernel2D = kernel @ kernel.T

print("Kernel 1D:\n", kernel)
print("Kernel 2D:\n", kernel2D)
print("Suma de los elementos del kernel 2D:", np.sum(kernel2D))

    # Aplicar suavizado gaussiano
img_suavizada = cv.filter2D(img, -1, kernel2D)
plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(img_suavizada, cmap='gray')
plt.title("Imagen con filtro Gaussiano")
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")

### OPCION 2 APLICARLO ###
# Aplicar filtro gaussiano
gauss = cv.GaussianBlur(img, (5, 5), 1)

# Mostrar resultado
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(gauss, cmap='gray')
plt.title("Gauss Blur")
plt.show()

'''
Se usa principalmente para suavizar o desenfocar imágenes de manera controlada y 
natural, reduciendo el ruido y los detalles finos.

Mayor tamaño de kernel = más desenfoque.
Si es muy grande, se pierde mucha información de la imagen (se ve muy borrosa).

La desviacion controla cuánto peso se le da a los píxeles vecinos.
Mayor sigma = la distribución gaussiana es más ancha, lo que también genera más suavizado.

¿Por qué usarlo en vez de promedio?
El filtro gaussiano respeta mejor las transiciones suaves y bordes.
Da más peso a los píxeles cercanos al centro del kernel (a diferencia del promedio, que trata todos por igual).
'''