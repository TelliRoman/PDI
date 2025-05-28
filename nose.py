import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------- FUNCIONES ---------
def filtro_alta_potencia(img, k=1.5):
    # Suavizado con filtro gaussiano
    suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    # Alta potencia: I + k*(I - suavizada)
    return cv2.addWeighted(img, 1 + k, suavizada, -k, 0)

def mostrar_imagenes(titulos, imagenes):
    plt.figure(figsize=(12, 6))
    for i in range(len(imagenes)):
        plt.subplot(1, len(imagenes), i + 1)
        plt.imshow(imagenes[i], cmap='gray')
        plt.title(titulos[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------- CARGA DE IMAGEN ---------
img = cv2.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\coins.tif', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("No se pudo cargar la imagen. Asegúrate de tener 'tu_imagen.jpg' en el mismo directorio.")

# --------- PROCESAMIENTO ---------
# Caso A: Ecualización → Alta potencia
img_ecualizada = cv2.equalizeHist(img)
resultado_A = filtro_alta_potencia(img_ecualizada)

# Caso B: Alta potencia → Ecualización
img_filtrada = filtro_alta_potencia(img)
resultado_B = cv2.equalizeHist(img_filtrada)

# --------- MOSTRAR RESULTADOS ---------
mostrar_imagenes(
    ["Original", "Ecualización → Alta potencia", "Alta potencia → Ecualización"],
    [img, resultado_A, resultado_B]
)

# --------- GUARDAR RESULTADOS (opcional) ---------
cv2.imwrite("resultado_ecualizacion_luego_filtro.jpg", resultado_A)
cv2.imwrite("resultado_filtro_luego_ecualizacion.jpg", resultado_B)
