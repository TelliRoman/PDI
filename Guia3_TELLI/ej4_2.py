import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

mariposa = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\mariposa02.png',cv.IMREAD_GRAYSCALE)
flores = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\flores02.jpg',cv.IMREAD_GRAYSCALE)
lapices = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\lapices02.jpg',cv.IMREAD_GRAYSCALE)
plt.figure(0)
plt.subplot(1,2,1)
mariposa_gaus = cv.GaussianBlur(mariposa, (5, 5), 1)
mariposa_suave = cv.boxFilter(mariposa,-1,(7,7),normalize=True)

plt.imshow(mariposa_suave, cmap='gray')
plt.title('Mariposa boxFilter 7x7')
plt.subplot(1,2,2)
mariposa_bilateral = cv.bilateralFilter(mariposa, 9, 75, 75)
plt.imshow(mariposa_bilateral, cmap='gray')
plt.title('Mariposa bilateral')


plt.figure(1)
plt.subplot(1,2,1)
flores_suave = cv.boxFilter(flores,-1,(7,7),normalize=True)
plt.imshow(flores_suave, cmap='gray')
plt.title('Flores boxFilter 7x7')
plt.subplot(1,2,2)
flores_bilateral = cv.bilateralFilter(flores, 9, 75, 75)
plt.imshow(flores_bilateral, cmap='gray')
plt.title('Flores bilateral')

plt.figure(2)
plt.subplot(1,2,1)
lapices_suave = cv.boxFilter(lapices,-1,(7,7),normalize=True)
plt.imshow(lapices_suave, cmap='gray')
plt.title('Lapices boxFilter 7x7')
plt.subplot(1,2,2)
lapices_bilateral = cv.bilateralFilter(lapices, 9, 75, 75)
plt.imshow(lapices_bilateral, cmap='gray')
plt.title('Lapices bilateral')

plt.show()

# Elegir una fila para el perfil (puede cambiarse)
fila = flores.shape[0] // 2  # Fila central de la imagen
# Obtener los perfiles
perfil_original = flores[fila, :]
perfil_box = flores_suave[fila, :]
perfil_bilateral = flores_bilateral[fila, :]

# Graficar los perfiles
plt.figure(figsize=(10, 5))
plt.plot(perfil_original, label='Original', color='black')
plt.plot(perfil_box, label='Box Filter', color='blue')
plt.plot(perfil_bilateral, label='Bilateral Filter', color='red')
plt.title(f'Perfiles de grises - Fila {fila}')
plt.xlabel('Columna')
plt.ylabel('Nivel de gris')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
El filtro bilateral es una técnica de suavizado no lineal que se utiliza para 
reducir el ruido en una imagen sin perder bordes importantes. 
Esto lo diferencia de otros filtros como el Gaussiano o el de mediana, 
que tienden a borrar los detalles en los bordes.'''

'''
Comparación | Box Filter | Filtro Bilateral
Suavizado general | Bueno, pero borra detalles finos | Bueno, mantiene más nitidez
Conservación de bordes | Baja | Alta (mantiene contornos bien definidos)
Rendimiento | Más rápido | Más lento (procesamiento no lineal)
Ideal para | Reducción básica de ruido | Procesamiento facial, bordes, arte'''

'''
En el gráfico se observan los perfiles de grises correspondientes a la misma fila de tres versiones de una imagen:

Perfil negro (Original): muestra variaciones abruptas de intensidad,
lo cual refleja la presencia de detalles y bordes marcados.

Perfil azul (Box Filter): presenta un perfil más suavizado con menor variación, 
lo que evidencia una reducción del ruido pero también una pérdida de detalles finos. El suavizado es uniforme y afecta tanto regiones planas como bordes.

Perfil rojo (Filtro Bilateral): logra una suavización efectiva sin alejarse 
demasiado del perfil original, especialmente en los bordes. 
Esto indica que preserva mejor los detalles mientras reduce el ruido en las 
zonas homogéneas.'''