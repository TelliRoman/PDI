import cv2 as cv
import numpy as np
import os

# Ruta a la carpeta con las imágenes de referencia
ruta = r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\Busqueda_histograma\Busqueda_histograma'
imagen_para_categorizar = r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\kikbut.jpg'
imagenes = []

# Obtener todos los archivos de imagen
archivos = [f for f in os.listdir(ruta) if f.lower().endswith(('.jpg', '.png', '.tif'))]
archivos.sort()

# Cargar las imágenes en escala de grises
for nombre in archivos:
    ruta_img = os.path.join(ruta, nombre)
    img = cv.imread(ruta_img, cv.IMREAD_GRAYSCALE)
    if img is not None:
        imagenes.append((nombre, img))  # Guardamos el nombre para saber la categoría

# Cargar imagen de consulta (externa)
img_consulta = cv.imread(imagen_para_categorizar, cv.IMREAD_GRAYSCALE)
hist_consulta = cv.calcHist([img_consulta], [0], None, [256], [0, 256])
cv.normalize(hist_consulta, hist_consulta)

# Lista para guardar (categoria, similitud)
similitudes = []

for nombre, img in imagenes:
    hist_img = cv.calcHist([img], [0], None, [256], [0, 256])
    cv.normalize(hist_img, hist_img)
    similitud = cv.compareHist(hist_consulta, hist_img, cv.HISTCMP_CORREL)

    # Inferir categoría desde el nombre del archivo
    nombre_lower = nombre.lower()
    if "bandera" in nombre_lower:
        categoria = "Bandera"
    elif "caricatura" in nombre_lower:
        categoria = "Caricatura"
    elif "personaje" in nombre_lower:
        categoria = "Personaje"
    elif "paisaje" in nombre_lower:
        categoria = "Paisaje"
    else:
        categoria = "Desconocida"

    similitudes.append((categoria, similitud))

# Agrupar similitudes por categoría
promedios = {}
conteos = {}

for categoria, sim in similitudes:
    if categoria not in promedios:
        promedios[categoria] = 0
        conteos[categoria] = 0
    promedios[categoria] += sim
    conteos[categoria] += 1

# Calcular el promedio de similitud por categoría
for categoria in promedios:
    promedios[categoria] /= conteos[categoria]

# Determinar la categoría más probable
categoria_max = max(promedios, key=promedios.get)

# Mostrar resultados
print(f"\nCategoría más probable según la imagen de consulta: {categoria_max}")
