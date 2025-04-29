import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

# Lectura de imágenes
img=cv.imread("imagenes/Deforestacion.png")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


### DETECCIÓN DE ZONA DE INTERÉS ###
# Cuantizar imagen para extraer valores blancos, lo demás queda negro
img_white = np.where(img_gray > 230, 255, 0)

# Obtener perfiles de intensidad de columnas y filas normalizados
row_intensity_profile = np.mean(img_white, axis=1)
row_intensity_min = np.min(row_intensity_profile)
row_intensity_profile -= row_intensity_min
row_intensity_max = np.max(row_intensity_profile)
row_intensity_profile = row_intensity_profile/row_intensity_max

col_intensity_profile = np.mean(img_white, axis=0)
col_intensity_min = np.min(col_intensity_profile)
col_intensity_profile -= col_intensity_min
col_intensity_max = np.max(col_intensity_profile)
col_intensity_profile = col_intensity_profile/col_intensity_max

# Cuantizar en dos valores para definir bordes
row_intensity_profile = np.pow(row_intensity_profile,3)
row_intensity_profile = np.where(row_intensity_profile > 0.5, 1, 0)

col_intensity_profile = np.pow(col_intensity_profile,3)
col_intensity_profile = np.where(col_intensity_profile > 0.5, 1, 0)

'''# Graficar perfiles de intensidad
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(row_intensity_profile)
plt.title("Perfil de Intensidad de Filas")
plt.xlabel("Fila")
plt.ylabel("Intensidad Promedio")

plt.subplot(1, 2, 2)
plt.plot(col_intensity_profile)
plt.title("Perfil de Intensidad de Columnas")
plt.xlabel("Columna")
plt.ylabel("Intensidad Promedio")

plt.tight_layout()
plt.show()'''

# Encontrar índices del perfil de intensidad donde el valor es 1 y el siguiente es 0
row_edges_10 = np.where((row_intensity_profile[:-1] == 1) & (row_intensity_profile[1:] == 0))[0]
col_edges_10 = np.where((col_intensity_profile[:-1] == 1) & (col_intensity_profile[1:] == 0))[0]

# Encontrar índices del perfil de intensidad donde el valor es 0 y el siguiente es 1
row_edges_01 = np.where((row_intensity_profile[:-1] == 0) & (row_intensity_profile[1:] == 1))[0]
col_edges_01 = np.where((col_intensity_profile[:-1] == 0) & (col_intensity_profile[1:] == 1))[0]

# Recortar la zona de interés
x_0 = col_edges_10[0]+1
x_1 = col_edges_01[-1]
y_0 = row_edges_10[0]+1
y_1 = row_edges_01[-1]
img_roi = img[y_0:y_1,x_0:x_1].copy()


### DETECCIÓN DE PÍXELES DE ESCALA ###
# Cuantizar imagen para extraer valores negros, lo demás queda blanco
img_black = np.where(img_gray < 5, 255, 0)

## Obtener perfiles de intensidad de filas normalizados
# Fijarse en las filas primero, para descartar la escala en millas
row_intensity_profile = np.mean(img_black, axis=1)
row_intensity_min = np.min(row_intensity_profile)
row_intensity_profile -= row_intensity_min
row_intensity_max = np.max(row_intensity_profile)
row_intensity_profile = row_intensity_profile/row_intensity_max

# Cuantizar en dos valores para definir bordes
row_intensity_profile = np.pow(row_intensity_profile,3)
row_intensity_profile = np.where(row_intensity_profile > 0.5, 1, 0)

# Encontrar índices del perfil de intensidad donde el valor es 0 y el siguiente es 1
row_edges_01 = np.where((row_intensity_profile[:-1] == 0) & (row_intensity_profile[1:] == 1))[0]

# Recortar imagen hasta la línea horizontal de la escala
img_black = img_black[0:row_edges_01[0],0:img_black.shape[1]].copy()

## Obtener perfiles de intensidad de filas normalizados
# Fijarse en las columnas ahora
col_intensity_profile = np.mean(img_black, axis=0)
col_intensity_min = np.min(col_intensity_profile)
col_intensity_profile -= col_intensity_min
col_intensity_max = np.max(col_intensity_profile)
col_intensity_profile = col_intensity_profile/col_intensity_max

# Cuantizar en dos valores para definir bordes
col_intensity_profile = np.pow(col_intensity_profile,3)
col_intensity_profile = np.where(col_intensity_profile > 0.5, 1, 0)

# Encontrar índices del perfil de intensidad donde el valor es 1 y el siguiente es 0
col_edges_10 = np.where((col_intensity_profile[:-1] == 1) & (col_intensity_profile[1:] == 0))[0]

# Encontrar índices del perfil de intensidad donde el valor es 0 y el siguiente es 1
col_edges_01 = np.where((col_intensity_profile[:-1] == 0) & (col_intensity_profile[1:] == 1))[0]

# 200 m en la imagen corresponden a scale_pixels
scale_length = col_edges_10[0]+1 - col_edges_01[-1]
pixel_length = 200 / scale_length
pixel_area = pixel_length**2

'''# Graficar perfiles de intensidad
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(row_intensity_profile)
plt.title("Perfil de Intensidad de Filas")
plt.xlabel("Fila")
plt.ylabel("Intensidad Promedio")

plt.subplot(1, 2, 2)
plt.plot(col_intensity_profile)
plt.title("Perfil de Intensidad de Columnas")
plt.xlabel("Columna")
plt.ylabel("Intensidad Promedio")

plt.tight_layout()
plt.show()'''


### VER ZONA DEFORESTADA ###
b,g,r = cv.split(img_roi)
h,s,v = cv.split(cv.cvtColor(img_roi,cv.COLOR_BGR2HSV))

# Aplicar un filtro de caja a la componente r
r = cv.boxFilter(r, ddepth=-1, ksize=(3, 3))

# Aplicar un filtro de mediana a la componente r
r = cv.medianBlur(r, 19)

hist_r = cv.calcHist([r], [0], None, [256], [0, 256])

# Graficar el histograma
'''plt.figure()
plt.title("Histograma en escala de grises")
plt.xlabel("Intensidad de píxeles")
plt.ylabel("Cantidad de píxeles")
plt.plot(hist_r)
plt.xlim([0, 256])
plt.show()'''

# Para r, se identifica un valle en la intensidad en 76, que puede representar el cambio en forestación
deforestacion = np.where(r > 76, 255, 0).astype(np.uint8)
deforestacion = cv.cvtColor(deforestacion.astype(np.uint8), cv.COLOR_GRAY2BGR)

# Colorear en la zona de interes en rojo la deforestación
img_roi[deforestacion[:,:,0] == 255] = [0, 0, 255]

### CALCULAR TAMAÑO DE LA ZONA DEFORESTADA ###
roi_area = img_roi.shape[0] * img_roi.shape[1] * pixel_area
deforestacion_area = np.sum(deforestacion[:,:,0] == 255) * pixel_area
monte_area = roi_area - deforestacion_area

# 1 ha = 10,000 m^2
print("Área de la zona de interés: ", roi_area/10000, "ha")
print("Área de la deforestación: ", deforestacion_area/10000, "ha")
print("Área de la zona con monte: ", monte_area/10000, "ha")

# Mostrar imágenes

# Imagen original
cv.imshow("img",img)

'''img_white = cv.cvtColor(img_white.astype(np.uint8), cv.COLOR_GRAY2BGR)
cv.imshow("img_white",img_white)'''

'''img_black = cv.cvtColor(img_black.astype(np.uint8), cv.COLOR_GRAY2BGR)
cv.imshow("img_black",img_black)'''

'''fig, ax = plt.subplots(nrows=2,ncols=3)
ax[0,0].imshow(b)
ax[0,1].imshow(g)
ax[0,2].imshow(r) 
ax[1,0].imshow(h)
ax[1,1].imshow(s)
ax[1,2].imshow(v)
plt.show()'''

# Imagen de la zona de interés
cv.imshow("img_roi",img_roi)

# Imagen de la deforestación
# cv.imshow("deforestacion",deforestacion)

cv.waitKey(0)
cv.destroyAllWindows()
