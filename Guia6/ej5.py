import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from filtros import filtro_rechazabanda_butterworth
from filtros import filtro_rechazabanda_ideal
from filtros import filtro_notch_ideal
from filtros import filtro_notch_butterworth
# Vector para almacenar las posiciones seleccionadas
posiciones = []

# Función callback para capturar eventos del mouse
def mostrar_posicion(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(f"Mouse en fila={y}, columna={x}")
        print(f"Valor de intensidad: {magnitude_spectrum[y, x]}")
        posiciones.append((x, y))

img = cv.imread(r'Imagenes\img_degradada.tif',cv.IMREAD_GRAYSCALE)
# Transformada de Fourier en 2D
f = np.fft.fft2(img)
# Desplazar el cero a la parte central
fshift = np.fft.fftshift(f)
# Espectro en magnitud (logarítmico para mejor visualización)
magnitude_spectrum = np.abs(fshift) #20 * np.log(np.abs(fshift) + 1)  # +1 para evitar log(0)

'''# Deteccion con clicks
# Mostrar imagen y asignar el callback
cv.namedWindow("Magnitude Spectrum")
cv.setMouseCallback("Magnitude Spectrum", mostrar_posicion)
# Redimensionar el espectro de magnitud para que ocupe más espacio en la ventana
#magnitude_spectrum = cv.resize(magnitude_spectrum, (600, 600), interpolation=cv.INTER_LINEAR)
while True:
    norm_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX)
    norm_spectrum = norm_spectrum.astype(np.uint8)
    cv.imshow('Magnitude Spectrum', norm_spectrum)
    #cv.resizeWindow("Magnitude Spectrum", 600, 600)
    if cv.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cv.destroyAllWindows()'''

# Deteccion Automatica

H = magnitude_spectrum.shape[0]
W = magnitude_spectrum.shape[1]
mask = np.ones(magnitude_spectrum.shape,magnitude_spectrum.dtype)
mask[int(H/3):int(2*H/3),int(W/3):int(2*W/3)] = 0 # Máscara rectangular
magnitude_spectrum_recorte = magnitude_spectrum * mask
# Encontrar coordenadas de los picos más relevantes en el espectro recortado
# Umbral para considerar solo los picos más altos
umbral = magnitude_spectrum_recorte.max() * 0.7
coords_picos = []
for y in range(magnitude_spectrum_recorte.shape[0]):  # filas
    for x in range(magnitude_spectrum_recorte.shape[1]):  # columnas
        if magnitude_spectrum_recorte[y, x] > umbral:
            coords_picos.append((x, y))  # (col, fila)

print("Coordenadas de los picos más relevantes:", coords_picos)

#Filtro rechaza banda ideal
# Crear filtro ideal
H = filtro_rechazabanda_ideal(magnitude_spectrum.shape, coords_picos, W=15)
plt.figure(3)
plt.imshow(H, cmap='gray')
for (x, y) in coords_picos:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Rechaza Banda Ideal')

# Aplicar al espectro
fshift_filtrado = fshift * H

# Inversa
f_ishift = np.fft.ifftshift(fshift_filtrado)
img_filtrada = np.fft.ifft2(f_ishift)
img_filtrada = np.abs(img_filtrada)

plt.figure(2,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada, cmap='gray')
plt.title('Filtrada con rechazabanda ideal')

#Filtro rechaza banda butterworth
# Crear filtro ideal
H1 = filtro_rechazabanda_butterworth(magnitude_spectrum.shape, coords_picos, W=15)
plt.figure(4)
plt.imshow(H1, cmap='gray')
for (x, y) in coords_picos:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Rechaza Banda butterworth')

# Aplicar al espectro
fshift_filtrado1 = fshift * H1

# Inversa
f_ishift1 = np.fft.ifftshift(fshift_filtrado1)
img_filtrada1 = np.fft.ifft2(f_ishift1)
img_filtrada1 = np.abs(img_filtrada1)

plt.figure(5,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada1, cmap='gray')
plt.title('Filtrada con rechazabanda butterworth')

#Filtro notch ideal
# Crear filtro ideal
H2 = filtro_notch_ideal(magnitude_spectrum.shape, coords_picos, 30)
plt.figure(6)
plt.imshow(H2, cmap='gray')
for (x, y) in coords_picos:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Notch ideal')

# Aplicar al espectro
fshift_filtrado2 = fshift * H2

# Inversa
f_ishift2 = np.fft.ifftshift(fshift_filtrado2)
img_filtrada2 = np.fft.ifft2(f_ishift2)
img_filtrada2 = np.abs(img_filtrada2)

plt.figure(7,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada2, cmap='gray')
plt.title('Filtrada Notch ideal')

#Filtro notch butterworth
# Crear filtro ideal
H3 = filtro_notch_butterworth(magnitude_spectrum.shape, coords_picos, 30,n=2)
plt.figure(8)
plt.imshow(H3, cmap='gray')
for (x, y) in coords_picos:
    plt.plot(x, y, 'ro')  # círculos rojos
plt.title('Filtro Notch Butterworth')

# Aplicar al espectro
fshift_filtrado3 = fshift * H3

# Inversa
f_ishift3 = np.fft.ifftshift(fshift_filtrado3)
img_filtrada3 = np.fft.ifft2(f_ishift3)
img_filtrada3 = np.abs(img_filtrada3)

plt.figure(9,figsize=(12,6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(img_filtrada3, cmap='gray')
plt.title('Filtrada Notch Butterworth')

#Mostrar Imagen degradada y su espectro
plt.figure(1,figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Degradada')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum.astype(float), cmap='gray')
plt.title('Espectro de Fourier')
'''
En el espectro de Fourier, vas a ver:
Un componente de baja frecuencia centrado (contenido de la imagen).
Pares de puntos brillantes (simétricos respecto al centro), que representan la interferencia sinusoidal.
El ruido sinusoidal se manifiesta como componentes de frecuencia alta aislados.
'''
plt.show()
