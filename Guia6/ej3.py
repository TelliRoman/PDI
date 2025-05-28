import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ej1_1 import ruido_exp
from ej1_1 import ruido_sp

def filtro_mediana(img,s,t): #Buena reducción de ruido impulsivo sin el desenfoque de un filtro lineal de la misma talla.
    img_sauv = np.zeros_like(img)
    pad_s = s // 2
    pad_t = t // 2
    padded = cv.copyMakeBorder(img, pad_s, pad_s, pad_t, pad_t, cv.BORDER_REPLICATE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            valores = []
            for m in range(-pad_s, pad_s + 1):
                for n in range(-pad_t, pad_t + 1):
                    valores.append(padded[i + pad_s + m, j + pad_t + n])
            mediana = np.median(valores)
            img_sauv[i, j] = mediana
    return img_sauv

def filtro_puntomedio(img,s,t):#Util para ruido tipo gaussiano o uniforme.
    img_sauv = np.zeros_like(img, dtype=np.float32)
    pad_s = s // 2
    pad_t = t // 2
    padded = cv.copyMakeBorder(img, pad_s, pad_s, pad_t, pad_t, cv.BORDER_REPLICATE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            valores = []
            for m in range(-pad_s, pad_s + 1):
                for n in range(-pad_t, pad_t + 1):
                    valores.append(padded[i + pad_s + m, j + pad_t + n])
            img_sauv[i, j] = 0.5 * (float(min(valores)) + float(max(valores)))
    return np.clip(img_sauv, 0, 255).astype(np.uint8)

#Util para combinaciones de ruido gaussiano y sal y pimienta.
def filtro_MediaAlfaRecortado(img, d, s, t):
    img_sauv = np.zeros_like(img, dtype=np.float32)
    pad_s = s // 2
    pad_t = t // 2
    padded = cv.copyMakeBorder(img, pad_s, pad_s, pad_t, pad_t, cv.BORDER_REPLICATE)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            valores = []
            for m in range(-pad_s, pad_s + 1):
                for n in range(-pad_t, pad_t + 1):
                    valor = float(padded[i + pad_s + m, j + pad_t + n])
                    valores.append(valor)
            valores.sort()
            d = min(d, len(valores) - 1)  # Evitar que d sea mayor que la cantidad de píxeles
            recorte = d // 2
            valores_recortados = valores[recorte:len(valores)-recorte]
            if len(valores_recortados) > 0:
                media = sum(valores_recortados) / len(valores_recortados)
            else:
                media = valores[len(valores) // 2]  # fallback
            img_sauv[i, j] = media
    
    return np.clip(img_sauv, 0, 255).astype(np.uint8)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
# Función para graficar histogramas
def graficar_histogramas(imagenes, titulos, fig_num):
    plt.figure(fig_num, figsize=(12, 6))
    for i, img in enumerate(imagenes):
        plt.subplot(1, len(imagenes), i+1)
        histima = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.bar(range(256), histima.ravel(), color='gray')
        plt.title(titulos[i])
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
    plt.tight_layout()

if __name__ == "__main__":
    img_sangre = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\sangre.jpg',cv.IMREAD_GRAYSCALE)

    ruido_gauss = cv.randn( np.zeros(img_sangre.shape, dtype=np.uint8) , 0, 20)
    img_sangre_ruido = img_sangre + ruido_gauss
    img_sangre_ruido = ruido_sp(img_sangre_ruido.copy(), 0.02, 0.02, 255, 0)

    img_mediana = filtro_mediana(img_sangre_ruido,3,3)
    img_punto_medio = filtro_puntomedio(img_sangre_ruido,3,3)
    img_mediaalfarecor = filtro_MediaAlfaRecortado(img_sangre,4,3,3)
    img_mediana_puntomedio = filtro_puntomedio(img_mediana,3,3)

    ecm_mediana = mse(img_sangre, img_mediana)
    ecm_puntomedio = mse(img_sangre, img_punto_medio)
    ecm_alfarecortado = mse(img_sangre, img_mediaalfarecor)
    ecm_medianapuntomedio = mse(img_sangre, img_mediana_puntomedio)
    ecms = [
        ("Filtro Mediana", ecm_mediana),
        ("Filtro Punto Medio", ecm_puntomedio),
        ("Filtro MediaAlfaRecortado", ecm_alfarecortado),
        ("Filtro Mediana + Punto Medio", ecm_medianapuntomedio)
    ]

    ecms_sorted = sorted(ecms, key=lambda x: x[1], reverse=True)

    graficar_histogramas(
    [img_sangre_ruido, img_mediana, img_punto_medio, img_mediaalfarecor, img_mediana_puntomedio],
    ["Ruido", "Mediana", "Punto Medio", "Media Alfa Recortado", "Mediana + Punto Medio"],
    fig_num=6
    )

    print("ECM de mayor a menor:")
    for nombre, valor in ecms_sorted:
        print(f"{nombre}: {valor:.2f}")

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(img_sangre, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Limpia')
    plt.subplot(1, 2, 2)
    plt.imshow(img_sangre_ruido, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(img_sangre_ruido, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')
    plt.subplot(1, 2, 2)
    plt.imshow(img_mediana, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtro Mediana')

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.imshow(img_sangre_ruido, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')
    plt.subplot(1, 2, 2)
    plt.imshow(img_punto_medio, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtro Punto Medio')

    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.imshow(img_sangre_ruido, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')
    plt.subplot(1, 2, 2)
    plt.imshow(img_mediaalfarecor, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtro MediaAlfaRecortado')

    plt.figure(5)
    plt.subplot(1, 2, 1)
    plt.imshow(img_sangre_ruido, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')
    plt.subplot(1, 2, 2)
    plt.imshow(img_mediana_puntomedio, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtro Mediana y Punto Medio')
    plt.show()

'''
Comparación cuantitativa (Error Cuadrático Medio - ECM):
ECM de mayor a menor:
Filtro Punto Medio: 1031.85
Filtro Mediana + Punto Medio: 260.37
Filtro Mediana: 135.57
Filtro MediaAlfaRecortado: 49.12

El filtro Media Alfa Recortado presenta el menor ECM, seguido de cerca la combinación Mediana 

2. Comparación cualitativa (subjetiva)
Filtro de Mediana: Elimina eficazmente el ruido impulsivo (sal y pimienta) sin generar tanto desenfoque como un filtro lineal. Sin embargo, es menos eficaz ante el ruido gaussiano, lo que puede dejar un grano visible en zonas homogéneas.

Filtro del Punto Medio: Promedia entre el mínimo y el máximo valor de la vecindad. Es adecuado para ruido gaussiano, pero poco eficaz contra ruido impulsivo, ya que puede verse afectado por los valores extremos (salt and pepper).

Filtro de Media Alfa Recortado: Elimina los valores extremos de la vecindad antes de calcular la media. Esto permite filtrar simultáneamente el ruido impulsivo y el gaussiano de forma balanceada. El resultado es más suave y cercano a la imagen original, sin tanto desenfoque ni artefactos visibles.

Mediana + Punto Medio: Primero elimina el ruido impulsivo y luego suaviza el ruido gaussiano restante. Esta combinación es efectiva, con resultados visuales satisfactorios y bajo ECM, aunque puede implicar un mayor costo computacional.

3. Observación de los histogramas
Los histogramas muestran cómo cada filtro afecta la distribución de niveles de gris:

El filtro de Mediana tiende a conservar la forma del histograma original pero reduciendo los picos extremos (valores 0 y 255).

El Punto Medio produce una concentración mayor en los valores medios, lo que sugiere un efecto suavizador.

El Media Alfa Recortado mantiene una distribución más balanceada, eliminando valores extremos y acercándose a la distribución original.

La combinación Mediana + Punto Medio aplana aún más los extremos y reduce la varianza del histograma, similar al efecto de un suavizado moderado.
'''
'''
El filtro de media geométrica no es adecuado para la imagen degradada con sal y pimienta, ya que el ruido impulsivo introduce ceros que inutilizan la media.

El filtro de media contra-armónica puede ser útil si se ajusta correctamente Q, pero no es robusto frente a una mezcla de sal y pimienta como la que se simuló.

La media alfa-recortada se posiciona como la opción más versátil, ya que no requiere conocer el tipo específico de ruido y funciona bien con ruido mixto.

'''