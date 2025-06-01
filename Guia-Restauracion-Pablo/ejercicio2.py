import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# === Filtros ===
def media_geometrica_manual(img, size=3):
    h, w = img.shape
    pad = size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            vecindad = img_padded[i:i+size, j:j+size].astype(np.float32)
            
            if np.any(vecindad <= 0):  # evitar ceros o negativos
                output[i, j] = 0
            else:
                log_vec = np.log(vecindad)
                media_log = np.mean(log_vec)
                output[i, j] = np.exp(media_log)

    return np.clip(output, 0, 255).astype(np.uint8)


def media_contra_armonica_manual(img, size=3, Q=0):
    h, w = img.shape
    pad = size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            vecindad = img_padded[i:i+size, j:j+size].astype(np.float32)
            if Q < 0:
                vecindad = vecindad[vecindad != 0]
                if vecindad.size == 0:
                    output[i, j] = 0
                    continue
            numerador = np.sum(vecindad ** (Q + 1))
            denominador = np.sum(vecindad ** Q)
            if denominador == 0:
                output[i, j] = 0
            else:
                output[i, j] = numerador / denominador

    return np.nan_to_num(np.clip(output, 0, 255)).astype(np.uint8)

# === Ruido ===
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

img_ruido=ruido_sal_pimienta(img_original, 0.1) #
img_ruido= ruido_gaussiano(img_ruido, 2)
img_ruido = np.clip(img_ruido, 0, 255).astype(np.uint8)

img_media_armonica = media_geometrica_manual(img_ruido, size=3)
img_media_contra_armonica = media_contra_armonica_manual(img_ruido, size=5, Q=0.3)

# === Mostrar ===
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_original, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_ruido, cmap='gray')
plt.title('Imagen con Ruido')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_media_armonica, cmap='gray')
plt.title('Media Geométrica')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_media_contra_armonica, cmap='gray')
plt.title('Media Contra Armónica')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(img_ruido.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
plt.title('Histograma Imagen con Ruido')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# === Cálculo del ECM ===
mse_armonica = mse(img_original, img_media_armonica)
mse_contra_armonica = mse(img_original, img_media_contra_armonica)

print(f"MSE Media Geometrica: {mse_armonica:.2f}")
print(f"MSE Media Contra Armónica: {mse_contra_armonica:.2f}")

## Es obvio que la media geometrica falle ya que hay ruido impulsivo y la media geometrica no es robusta a este tipo de ruido.
