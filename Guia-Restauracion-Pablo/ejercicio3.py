import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def filtro_mediana(img, size=3):
    return cv.medianBlur(img, size)

def filtro_punto_medio(img, size=3):
    h, w = img.shape
    pad = size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            vecindad = img_padded[i:i+size, j:j+size]
            output[i, j] = (float(np.min(vecindad)) + float(np.max(vecindad))) / 2.0


    return np.clip(output, 0, 255).astype(np.uint8)

def filtro_media_alfa_recortado(img, size=3, d=2):
    h, w = img.shape
    pad = size // 2
    n = size * size
    img_padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            vec = img_padded[i:i+size, j:j+size].flatten()
            vec_ordenado = np.sort(vec)
            if d >= n:
                output[i, j] = 0
            else:
                recortado = vec_ordenado[d//2 : n - d//2]
                output[i, j] = np.mean(recortado)

    return np.clip(output, 0, 255).astype(np.uint8)

def filtro_mediana_punto_medio(img, size=3):
    img_mediana = filtro_mediana(img, size)
    return filtro_punto_medio(img_mediana, size)

# === Ruido ===
def ruido_impulso_unimodal(img, prob):
    noisy_img = img.copy()
    mask = np.random.rand(*img.shape) < prob
    random_vals = np.random.randint(0, 256, img.shape)
    noisy_img[mask] = random_vals[mask]
    return noisy_img
def ruido_sal_pimienta(img, prob): #prob: probabilidad total de que un píxel se corrompa (mitad sal, mitad pimienta)
    noisy_img = img.copy()
    rand = np.random.rand(*img.shape)
    noisy_img[rand < prob / 2] = 0      # Pimienta
    noisy_img[rand > 1 - prob / 2] = 255 # Sal
    return noisy_img
def ruido_gaussiano(img, std_dev):
    noise = np.random.normal(0, std_dev, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# === ECM ===
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# === Procesamiento ===
img_original = cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\sangre.jpg", cv.IMREAD_GRAYSCALE)

img_ruido = ruido_sal_pimienta(img_original, 0.1) #
img_ruido= ruido_gaussiano(img_ruido, 20)
img_ruido = np.clip(img_ruido, 0, 255).astype(np.uint8)

# Aplicar filtros
img_mediana = filtro_mediana(img_ruido, size=3)
img_punto_medio = filtro_punto_medio(img_ruido, size=5)
img_alfa_recortado = filtro_media_alfa_recortado(img_ruido, size=3, d=2)
img_mediana_punto = filtro_mediana_punto_medio(img_ruido, size=3)


# Mostrar imágenes
plt.figure(figsize=(12, 8)) 
plt.suptitle("Imágenes filtradas", fontsize=16)

plt.subplot(2, 3, 1)
plt.imshow(img_original, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_ruido, cmap='gray')
plt.title("Con Ruido")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_mediana, cmap='gray')
plt.title("Filtro Mediana")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_punto_medio, cmap='gray')
plt.title("Filtro Punto Medio")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_alfa_recortado, cmap='gray')
plt.title("Media-Alfa Recortado")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_mediana_punto, cmap='gray')
plt.title("Mediana + Punto Medio")
plt.axis('off')

plt.tight_layout()
plt.show()



## Estos filtro funcionan mejor q