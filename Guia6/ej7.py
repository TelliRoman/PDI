import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ej2_1 import *
from ej3_1 import *

def encontrar_bloque_mas_homogeneo(img, block_size=32):
    #Inizializar varianza en un valor muy alto y el bloque más homogéneo en (0, 0).
    min_var = float('inf')
    mejor_bloque = (0, 0)
    for y in range(0, img.shape[0] - block_size + 1, block_size):
        for x in range(0, img.shape[1] - block_size + 1, block_size):
            #Extraer el bloque y calcular su varianza.
            bloque = img[y:y+block_size, x:x+block_size]
            var = np.var(bloque)
            #Si la varianza del bloque es menor que la mínima encontrada, actualizar el bloque más homogéneo.
            if var < min_var:
                min_var = var
                mejor_bloque = (x, y)
    return mejor_bloque, block_size

#Leer imagenes
img_a = cv.imread(r'Imagenes\FAMILIA_a.jpg',cv.IMREAD_GRAYSCALE)
img_b = cv.imread(r'Imagenes\FAMILIA_b.jpg',cv.IMREAD_GRAYSCALE)
img_c = cv.imread(r'Imagenes\FAMILIA_c.jpg',cv.IMREAD_GRAYSCALE)
#Recortar áreas constantes en imágenes adquiridas.
(x, y), block_size = encontrar_bloque_mas_homogeneo(img_a, block_size=128)
#Extraer bloques homogéneos de las imágenes.
bloque_a = img_a[y:y+block_size, x:x+block_size]
bloque_b = img_b[y:y+block_size, x:x+block_size]
bloque_c = img_c[y:y+block_size, x:x+block_size]
#Dibujar rectángulos en las imágenes originales para mostrar el área homogénea.
img_a_recthomogeneo =cv.rectangle(img_a.copy(),(x,y),(x+block_size,y+block_size),(0,0,0),2)
img_b_recthomogeneo =cv.rectangle(img_b.copy(),(x,y),(x+block_size,y+block_size),(0,0,0),2)
img_c_recthomogeneo =cv.rectangle(img_c.copy(),(x,y),(x+block_size,y+block_size),(0,0,0),2)

plt.figure(0)
plt.subplot(121)
plt.imshow(img_a_recthomogeneo, cmap='gray', vmin=0, vmax=255)
plt.subplot(122)
plt.imshow(bloque_a, cmap='gray', vmin=0, vmax=255)
hist_a = cv.calcHist([bloque_a], [0], None, [256], [0, 256])
hist_b = cv.calcHist([bloque_b], [0], None, [256], [0, 256])
hist_c = cv.calcHist([bloque_c], [0], None, [256], [0, 256])

plt.figure(1)
plt.subplot(121)
plt.imshow(img_a_recthomogeneo, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen A')
plt.subplot(122)
plt.bar(range(256), hist_a.ravel(), color='gray')
plt.title('Histograma A')

plt.figure(2)
plt.subplot(121)
plt.imshow(img_b_recthomogeneo, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen B')
plt.subplot(122)
plt.bar(range(256), hist_b.ravel(), color='gray')
plt.title('Histograma B')

plt.figure(3)
plt.subplot(121)
plt.imshow(img_c_recthomogeneo, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen C')
plt.subplot(122)
plt.bar(range(256), hist_c.ravel(), color='gray')
plt.title('Histograma C')
plt.show()
#Por observacion de los histogramas de las areas mas homogeneas de la imagenes se puede ver que:
# -La imagen A tiene un ruido gaussiano aditivo, ya que se forma una campana de gauss al rededor del pico
# -La imagen B tiene un ruido uniforme, ya que el histograma es plano y no tiene picos definidos
# -La imagen C tiene un ruido de sal y pimienta, ya que el histograma tiene picos en los extremos (0 y 255)

#calcular los parametros estadisticos para dichos ruidos
media_a = np.mean(bloque_a)
varianza_a = np.var(bloque_a)
desvio_a = np.std(bloque_a)
print(f"Media estimada de A: {media_a}")
#El desvío estándar es la raíz cuadrada de la varianza. Ambos describen la dispersión del ruido gaussiano, pero el desvío estándar está en las mismas unidades que la imagen.
print(f"Varianza estimada de A: {varianza_a}")
print(f"Desvio estimado de A: {desvio_a}")

a = float(bloque_b.min())
b = float(bloque_b.max())
media_b = (a + b) / 2.0
varianza_b = ((b - a) ** 2.0) / 12.0
print(f"Media estimada de B: {media_b}")
print(f"Varianza estimada de B: {varianza_b}")

#Analizar si la sal y pimienta son 0 y 255 en el histograma
# Rango de búsqueda para pimienta y sal (ajusta según tu caso)
rango_pimienta = hist_c[:30]   # Intensidades bajas
rango_sal = hist_c[225:]       # Intensidades altas

# Índices de los máximos en cada rango
pico_pimienta = np.argmax(rango_pimienta)
pico_sal = 225 + np.argmax(rango_sal)

print(f"Pico de pimienta (intensidad baja): {pico_pimienta}")
print(f"Pico de sal (intensidad alta): {pico_sal}")

total = bloque_c.size
Pa = np.sum(bloque_c == pico_pimienta) / total
Pb = np.sum(bloque_c == pico_sal) / total
print(f"Probabilidad de 0 (Pa): {Pa}")
print(f"Probabilidad de 255 (Pb): {Pb}")

#Elegir los filtros mas adecuados para cada tipo de ruido
#Para el rudio gaussiano usar filtro de media geometrica o armonica
#Para el ruido uniforme usar filtro de puntomedio
#Para el ruido de sal y pimienta usar filtro de mediana

img_a_filtrada = filtro_media_geom_log(img_a,3,3)
#img_a_filtrada = filtro_contraarmonica(img_a,-1,3,3)
img_b_filtrada = filtro_puntomedio(img_b,3,3)
#img_c_filtrada = filtro_mediana(img_c,3,3)
img_c_filtrada = cv.medianBlur(img_c,3)

plt.figure(1)
plt.subplot(121)
plt.imshow(img_a, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen A original')
plt.subplot(122)
plt.imshow(img_a_filtrada, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen A Filtrada')

plt.figure(2)
plt.subplot(121)
plt.imshow(img_b, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen B original')
plt.subplot(122)
plt.imshow(img_b_filtrada, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen B Filtrada')

plt.figure(3)
plt.subplot(121)
plt.imshow(img_c, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen C original')
plt.subplot(122)
plt.imshow(img_c_filtrada, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen C Filtrada')
plt.show()
