import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from filtros import filtro_rechazabanda_butterworth
from filtros import filtro_rechazabanda_ideal
from filtros import filtro_notch_ideal
from filtros import filtro_notch_butterworth
from filtros import mse

def deteccion_picos(magnitude_spectrum):
    # Deteccion Automatica
    H = magnitude_spectrum.shape[0]
    W = magnitude_spectrum.shape[1]
    mask = np.ones(magnitude_spectrum.shape,magnitude_spectrum.dtype)
    #Usar esta mascara para la imagen de la luna
    mask[int(H/3):int(2*H/3),int(W/3):int(2*W/3)] = 0 # Máscara rectangular

    # Crear máscara rectangular centrada con ceros en 9 píxeles a izquierda/derecha y 19 arriba/abajo del centro
    # Usar esta mascara para la imagen del craneo
    #center_y, center_x = H // 2, W // 2
    #mask[max(0, center_y - 19):min(H, center_y + 20), max(0, center_x - 9):min(W, center_x + 10)] = 0

    magnitude_spectrum_recorte = magnitude_spectrum * mask
    # Encontrar coordenadas de los picos más relevantes en el espectro recortado
    # Umbral para considerar solo los picos más altos
    umbral = magnitude_spectrum_recorte.max() * 0.8
    coords_picos = []
    visitado = np.zeros_like(magnitude_spectrum, dtype=bool)
    # Definimos el tamaño de vecindad para evitar picos sucesivos (por ejemplo, 5x5)
    vecindad = 5
    for y in range(magnitude_spectrum_recorte.shape[0]):  # filas
        for x in range(magnitude_spectrum_recorte.shape[1]):  # columnas
            if magnitude_spectrum_recorte[y, x] > umbral and not visitado[y, x]:
                coords_picos.append((x, y))
                # Marcamos como visitados los vecinos cercanos para evitar duplicados
                y_min = max(0, y - vecindad // 2)
                y_max = min(magnitude_spectrum.shape[0], y + vecindad // 2 + 1)
                x_min = max(0, x - vecindad // 2)
                x_max = min(magnitude_spectrum.shape[1], x + vecindad // 2 + 1)
                visitado[y_min:y_max, x_min:x_max] = True
    return coords_picos

#NOMBRAR IMG1 A LA QUE SE QUIERE VISUALIZAR Y ACORDARSE DE CAMBIAR LA MASK EN LADETECCION DE PICOS
img1 = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\noisy_moon.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\HeadCT_degradada.tif',cv.IMREAD_GRAYSCALE)

# Transformada de Fourier en 2D
f1 = np.fft.fft2(img1)
# Desplazar el cero a la parte central
fshift1 = np.fft.fftshift(f1)
# Espectro en magnitud (logarítmico para mejor visualización)
magnitude_spectrum1 = np.abs(fshift1) #20 * np.log(np.abs(fshift) + 1)  # +1 para evitar log(0)

f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = np.abs(fshift2) #20 * np.log(np.abs(fshift) + 1)  # +1 para evitar log(0)

#Mostrar Imagen degradada y su espectro
plt.figure(1,figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Imagen 1 Degradada')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum1.astype(float), cmap='gray')
plt.title('Espectro de Fourier 1')

coords1 = deteccion_picos(magnitude_spectrum1.copy())
#coords2 = deteccion_picos(magnitude_spectrum2.copy())

print("Coordenadas de los picos más relevantes img 1:", coords1)
#print("Coordenadas de los picos más relevantes img 2:", coords2)

#Filtro rechaza banda butterworth
# Crear filtro ideal
H1 = filtro_rechazabanda_butterworth(magnitude_spectrum1.shape, coords1, W=10)
plt.figure(3)
plt.imshow(H1, cmap='gray')
for (x, y) in coords1:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Rechaza Banda butterworth')

# Aplicar al espectro
fshift_filtrado1 = fshift1 * H1

# Inversa
f_ishift1 = np.fft.ifftshift(fshift_filtrado1)
img_filtrada1 = np.fft.ifft2(f_ishift1)
img_filtrada1 = np.abs(img_filtrada1)

plt.figure(4,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada1, cmap='gray')
plt.title('Filtrada con rechazabanda butterworth')

#Filtro notch butterworth
# Crear filtro ideal
H2 = filtro_notch_butterworth(magnitude_spectrum1.shape, coords1, 10,n=2)
plt.figure(5)
plt.imshow(H2, cmap='gray')
for (x, y) in coords1:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Notch Butterworth')

# Aplicar al espectro
fshift_filtrado2 = fshift1 * H2

# Inversa
f_ishift2 = np.fft.ifftshift(fshift_filtrado2)
img_filtrada2 = np.fft.ifft2(f_ishift2)
img_filtrada2 = np.abs(img_filtrada2)

plt.figure(9,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada2, cmap='gray')
plt.title('Filtrada Notch Butterworth')

'''
EN la imagen de la luna usar un W=10 o 15 como parametro de los filtros
EN la imagen del craneo usar un W=3 o 5 como parametro de los filtros ya que el ruido senoidal es 
de baja frecuencia y un ancho del filtro muy grande no deja pasar lasfrecuenciasde la imagen
'''
plt.show()