import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\futbol.jpg",cv.IMREAD_GRAYSCALE)
img_float = np.float32(img)

# 2. Calcular la DFT
dft = cv.dft(img_float, flags=cv.DFT_COMPLEX_OUTPUT)

# 3. Separar parte real e imaginaria
Re, Im = cv.split(dft)

# 4. Obtener magnitud y fase
magnitude, phase = cv.cartToPolar(Re, Im)

# 5. Reconstruir con solo magnitud (fase = 0)
Re_mag_only = magnitude * np.cos(0)
Im_mag_only = magnitude * np.sin(0)
dft_mag_only = cv.merge([Re_mag_only, Im_mag_only])
img_mag_only = cv.idft(dft_mag_only, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

# 6. Reconstruir con solo fase (magnitud = 1)
Re_phase_only = np.cos(phase)
Im_phase_only = np.sin(phase)
dft_phase_only = cv.merge([Re_phase_only, Im_phase_only])
img_phase_only = cv.idft(dft_phase_only, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

# 7. Normalizar para visualizar
img_recon = cv.idft(dft, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_recon = cv.normalize(img_recon, None, 0, 255, cv.NORM_MINMAX)
img_mag_only = cv.normalize(img_mag_only, None, 0, 255, cv.NORM_MINMAX)
img_phase_only = cv.normalize(img_phase_only, None, 0, 255, cv.NORM_MINMAX)

# 8. Mostrar resultados
plt.figure(figsize=(10, 6))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(img_recon, cmap='gray')
plt.title("Reconstruida Completa")

plt.subplot(1, 4, 3)
plt.imshow(img_mag_only, cmap='gray')
plt.title("Solo Magnitud")

plt.subplot(1, 4, 4)
plt.imshow(img_phase_only, cmap='gray')
plt.title("Solo Fase")

plt.tight_layout()
plt.show()


#Imagen reconstruida solo con magnitud:
#Muy difusa, borrosa o sin detalles definidos. Conserva información de energía pero no contiene estructuras claras.

#Imagen reconstruida solo con fase:
#Sorprendentemente parecida a la original. Aunque con bajo contraste, preserva completamente la forma y estructura.