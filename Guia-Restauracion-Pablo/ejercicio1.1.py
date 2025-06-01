import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def ruido_gaussiano(img, std_dev): #desviación estándar del ruido (más alto = más ruido)
    noise = np.random.normal(0, std_dev, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def ruido_uniforme(img, intensity): #máximo valor absoluto del ruido uniforme (ruido ∈ [−intensity, +intensity])
    noise = np.random.uniform(-intensity, intensity, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def ruido_sal_pimienta(img, prob): #prob: probabilidad total de que un píxel se corrompa (mitad sal, mitad pimienta)
    noisy_img = img.copy()
    rand = np.random.rand(*img.shape)
    noisy_img[rand < prob / 2] = 0      # Pimienta
    noisy_img[rand > 1 - prob / 2] = 255 # Sal
    return noisy_img

def ruido_impulso_unimodal(img, prob): # probabilidad de que un píxel se reemplace por un valor aleatorio
    noisy_img = img.copy()
    mask = np.random.rand(*img.shape) < prob
    random_vals = np.random.randint(0, 256, img.shape)
    noisy_img[mask] = random_vals[mask]
    return noisy_img

def ruido_exponencial(img, scale):#parámetro de escala de la distribución exponencial (media = scale, se centra luego en 0)
    noise = np.random.exponential(scale, img.shape)
    noise = noise - np.mean(noise)  # centrar en 0
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\futbol.jpg", cv.IMREAD_GRAYSCALE)
ruido_exponencial_img = ruido_impulso_unimodal(img, 0.1)


# Mostrar imagen original, imagen con ruido y sus histogramas
plt.figure(figsize=(12, 8))

# Imagen original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Imagen con ruido
plt.subplot(2, 2, 2)
plt.imshow(ruido_exponencial_img, cmap='gray')
plt.title('Imagen con Ruido ')
plt.axis('off')

# Histograma original
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
plt.title('Histograma Imagen Original')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

# Histograma con ruido
plt.subplot(2, 2, 4)
plt.hist(ruido_exponencial_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
plt.title('Histograma Imagen con Ruido Exponencial')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()
