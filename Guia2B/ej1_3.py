import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\coin.jpg', cv.IMREAD_GRAYSCALE)

# Parámetro A (>1 para high-boost)
A = 1.2# Podés ajustar este valor

# Suavizado con filtro promedio (puede ser también Gaussiano)
blurred = cv.GaussianBlur(img, (5 ,5), 0)

# Calcular imagen high-boost
img_highboost = cv.addWeighted(src1=img, alpha=A, src2=blurred, beta=-1, gamma=0)

# Calcular histogramas
hist_orig = cv.calcHist([img], [0], None, [256], [0,256])
hist_highboost = cv.calcHist([img_highboost], [0], None, [256], [0,256])

# Mostrar todo en una figura
plt.figure()

# Imagen original
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")
plt.axis('off')

# Histograma original
plt.subplot(2,2,2)
plt.plot(hist_orig, color='black')
plt.title("Histograma original")
plt.xlim([0,256])
plt.ylim([0,1200])

# Imagen high-boost
plt.subplot(2,2,3)
plt.imshow(img_highboost, cmap='gray')
plt.title(f"Filtro High-Boost (A={A})")
plt.axis('off')

# Histograma high-boost
plt.subplot(2,2,4)
plt.plot(hist_highboost, color='black')
plt.title("Histograma high-boost")
plt.xlim([0,256])
plt.ylim([0,1200])

plt.tight_layout()
plt.show()
