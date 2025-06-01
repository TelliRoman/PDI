import numpy as np
import cv2
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

# Crear imagen 600x600 en blanco (uint8)
img = np.zeros((600, 600), dtype=np.uint8)
# Franja izquierda (gris claro ≈ 180)
img[:, :200] = 180
# Franja central (gris medio ≈ 120)
img[:, 200:400] = 120
# Franja derecha (gris oscuro ≈ 60)
img[:, 400:] = 60
# Mostrar imagen y su histograma
plt.figure(figsize=(10, 5))


img_ruido = ruido_gaussiano(img, 10)  # Aplicar ruido gaussiano con desviación estándar de 20
# Imagen
plt.subplot(1, 2, 1)
plt.imshow(img_ruido, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen con 3 franjas')
plt.axis('off')

# Histograma
plt.subplot(1, 2, 2)
plt.hist(img_ruido.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.8)
plt.title('Histograma')
plt.xlabel('Intensidad de gris')
plt.ylabel('Cantidad de píxeles')

plt.tight_layout()
plt.show()
