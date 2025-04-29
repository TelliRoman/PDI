import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def calcular_estadisticas_desde_imagen(imagen):
    # Asegurarse de que sea tipo float para evitar errores en cálculos
    img = imagen.astype(np.float64)

    # Normalizar valores para obtener probabilidades
    total_pixeles = img.size
    valores, cuentas = np.unique(img, return_counts=True)
    probs = cuentas / total_pixeles

    # Media
    media = np.mean(img)

    # Varianza
    varianza = np.var(img)

    # Asimetría (skewness)
    asimetria = np.mean(((img - media) ** 3)) / (np.std(img) ** 3 + 1e-7)  # evitar división por cero

    # Energía: suma de los cuadrados de las probabilidades
    energia = np.sum(probs ** 2)

    # Entropía: mide el desorden
    entropia = -np.sum(probs * np.log2(probs + 1e-7))  # evitar log(0)

    

    return {
        "Media": media,
        "Varianza": varianza,
        "Asimetría": asimetria,
        "Energía": energia,
        "Entropía": entropia
    }


# =============================================
# Parte 1: Análisis de patrones
# =============================================

# Cargar imágenes de patrones
patron1 = cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\patron2.tif")
patron2 = cv.imread(
    r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\patron.tif", 
    cv.IMREAD_GRAYSCALE
)

# Generar y mostrar histograma para patron1 (RGB)
histr = cv.calcHist([patron1], [0], None, [256], [0, 256])
plt.plot(histr, color="r")
plt.xlim([0, 256])
plt.title("Histograma")
plt.xlabel("Nivel de Intensidad")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# =============================================
# Parte 2: Análisis de imágenes (A, B, C, D)
# =============================================

def mostrar_histograma(imagen, titulo):
    hist = cv.calcHist([imagen], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(titulo)
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.xlabel("Nivel de Intensidad")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

# Cargar imágenes en escala de grises
imagenA = cv.imread(
    r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\imagenA.tif", 
    cv.IMREAD_GRAYSCALE
)
imagenB = cv.imread(
    r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\imagenB.tif", 
    cv.IMREAD_GRAYSCALE
)
imagenC = cv.imread(
    r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\imagenC.tif", 
    cv.IMREAD_GRAYSCALE
)
imagenD = cv.imread(
    r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\imagenD.tif", 
    cv.IMREAD_GRAYSCALE
)

# Mostrar histogramas
mostrar_histograma(imagenA, "Histograma - Imagen A")
mostrar_histograma(imagenB, "Histograma - Imagen B")
mostrar_histograma(imagenC, "Histograma - Imagen C")
mostrar_histograma(imagenD, "Histograma - Imagen D")


# Media: indica el brillo promedio de la imagen. Valores altos → imagen más clara; valores bajos → más oscura.

# Varianza: mide la dispersión de los niveles de intensidad. Alta varianza → imagen con mucho contraste.

# Asimetría (skewness): indica si la distribución está sesgada hacia los tonos oscuros (asimetría positiva) o claros (negativa).

# Energía: mide la uniformidad de la imagen. Alta energía → pocos niveles de gris dominan (imagen más uniforme o con menos detalle).

# Entropía: mide el desorden o la complejidad de la imagen. Alta entropía → mucha variación de intensidades (imagen rica en detalles o ruido).
