import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

print("Python version %s / OpenCV version %s " %(sys.version,cv.__version__))

# Ustedes deberán indicar el path a la imagen en su Drive o subir el archivo correspondiente a Colab.

img_gray = cv.imread('Guia1/camino.tif',cv.IMREAD_GRAYSCALE)
img_color = cv.imread('Guia1/camino.tif') # color

# Visualización con Opencv: esto no funciona en el entorno Colab, pero es util
# para hacer aplicaciones en PC.
# cv.imshow("Titulo 1",img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

plt.imshow(img_gray)

print("Valores de una esquina de la imagen (2x2 pixeles):")
print(img_color[:2, :2, :])
print()

print("Alto, ancho y n_canales = %s" %str(img_color.shape))
print("Tipo de datos de imagen", img_color.dtype)

print("valor medio %.3f, mínimo %d, máximo %d" %(np.mean(img_color),
                                                 np.min(img_color),
                                                 np.max(img_color)))

# Podemos generar una mascara binaria para operar con imágenes.
mask = np.zeros(img_gray.shape, dtype=img_gray.dtype) # dtype: 'int8', bool, float, tener en cuenta el tipo de dato al realizar las operaciones

# O bien:
mask = img_gray.copy()
mask[:]=0

# Las imagenes son objetos y para hacer copias se debe tener en cuenta que el operador de asignación ("=") copia el puntero al objeto.
# Generamos una ROI con unos en la máscara
H, W = img_gray.shape
mask[int(H/3):int(2*H/3),int(W/3):int(2*W/3)]=255 # Máscara rectangular

# operación AND pixel a pixel
img_recorte=cv.bitwise_and(img_gray,mask)

plt.imshow(img_recorte,cmap="gray",vmin=0,vmax=255)
plt.show()

line=200
plt.figure()
plt.imshow(img_gray,cmap='gray')
plt.plot([0,img_gray.shape[1]],[line,line])

plt.figure()
plt.plot(img_gray[line,:])
plt.title("Perfil de intensidad")

plt.figure()
#cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
hist=cv.calcHist([img_gray], [0], None, [256], [0,256])
plt.bar(range(256),np.squeeze(hist))
plt.title("Histograma de la imagen")

plt.show()

