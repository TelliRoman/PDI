import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Paso 1: leer e ir a float
img_ferrari = cv.imread(r"C:\Users\pablo\OneDrive\Desktop\PDI\Imagenes\ferrari-c.jpg", cv.IMREAD_GRAYSCALE)
img_puente = cv.imread(r"C:\Users\pablo\OneDrive\Desktop\PDI\Imagenes\puente.jpg", cv.IMREAD_GRAYSCALE)
img_ferrari = img_ferrari[:900, :900]
img_puente = img_puente[:900, :900]

img_ferrari_float = np.float32(img_ferrari)
img_puente_float = np.float32(img_puente)

# Paso 2: DFT
dft_ferrari = cv.dft(img_ferrari_float, flags=cv.DFT_COMPLEX_OUTPUT)
dft_puente = cv.dft(img_puente_float, flags=cv.DFT_COMPLEX_OUTPUT)

# Paso 3: separar real e imaginario
Re_ferrari, Im_ferrari = cv.split(dft_ferrari)
Re_puente, Im_puente = cv.split(dft_puente)

# Paso 4: obtener magnitud y fase
mag_ferrari, phase_ferrari = cv.cartToPolar(Re_ferrari, Im_ferrari)
mag_puente, phase_puente = cv.cartToPolar(Re_puente, Im_puente)

# Paso 5: combinar cruces (mag A + fase B y mag B + fase A)
Re_fA_pB = mag_ferrari * np.cos(phase_puente)
Im_fA_pB = mag_ferrari * np.sin(phase_puente)
dft_fA_pB = cv.merge([Re_fA_pB, Im_fA_pB])

Re_fB_pA = mag_puente * np.cos(phase_ferrari)
Im_fB_pA = mag_puente * np.sin(phase_ferrari)
dft_fB_pA = cv.merge([Re_fB_pA, Im_fB_pA])

# Paso 6: reconstrucción con IDFT
img_fA_pB = cv.idft(dft_fA_pB, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
img_fB_pA = cv.idft(dft_fB_pA, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

# Paso 7: normalizar para mostrar
img_fA_pB = cv.normalize(img_fA_pB, None, 0, 255, cv.NORM_MINMAX)
img_fB_pA = cv.normalize(img_fB_pA, None, 0, 255, cv.NORM_MINMAX)

# Paso 8: mostrar
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(img_ferrari, cmap='gray')
plt.title("Original Ferrari")

plt.subplot(2, 2, 2)
plt.imshow(img_puente, cmap='gray')
plt.title("Original Puente")

plt.subplot(2, 2, 3)
plt.imshow(img_fA_pB.astype(np.uint8), cmap='gray')
plt.title("Magnitud Ferrari + Fase Puente")

plt.subplot(2, 2, 4)
plt.imshow(img_fB_pA.astype(np.uint8), cmap='gray')
plt.title("Magnitud Puente + Fase Ferrari")

plt.tight_layout()
plt.show()
