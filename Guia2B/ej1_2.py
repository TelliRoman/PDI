import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

ruta_base = 'C:\\Users\\pablo\\Desktop\\PDI\\PDI\\Imagenes\\'  
histo  = []
for i in range(1, 6):
    nombre_archivo = f'histo{i}.tif'
    path_completo = ruta_base + nombre_archivo
    imagen = cv.imread(path_completo, cv.IMREAD_GRAYSCALE)
    
    if imagen is not None:
        histo.append(imagen)
    else:
        print(f'No se pudo cargar {path_completo}')
for i in range(0,5):
    plt.figure(i)
    plt.imshow(histo[i], cmap='gray')

#Histograma 1 tiene un pico en las intensidades bajas por lo que es mas oscura,
#y todas las otras intensidades de grises tienen valores equilibrados por lo que tiene buen contraste
#Histograma 2 es media oscura (nivel de gris cercano a 100) y con poco constraste
#Histograma 3 es muy oscura y con poco contraste
#Histograma 4 es muy clara y con poco contraste
#Histograma 5 es clara y con contraste medio
imagenes  = []
nombres = ['A', 'B', 'C', 'D', 'E']  # Letras que identifican las imágenes

for letra in nombres:
    nombre_archivo = f'imagen{letra}.tif'
    path_completo = ruta_base + nombre_archivo
    imagen = cv.imread(path_completo, cv.IMREAD_GRAYSCALE)
    
    if imagen is not None:
        imagenes.append(imagen)
    else:
        print(f'No se pudo cargar {path_completo}')

for i in range(len(imagenes)):
    plt.figure(i+5)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(f'Imagen {nombres[i]}')

#A-2
#B-4
#C-1       
#D-5
#E-3

for i in range(len(imagenes)):
    histo = cv.calcHist([imagenes[i]], [0], None, [256], [0, 256])
    hist_norm = histo / histo.sum()
    niveles = np.arange(256)

    media = np.sum(niveles * hist_norm[:,0])
    varianza = np.sum(((niveles - media) ** 2) * hist_norm[:, 0])
    asimetria = np.sum(((niveles - media) ** 3) * hist_norm[:, 0]) / (varianza ** 1.5)
    energia = np.sum(hist_norm[:, 0] ** 2)
    # Evitar log(0) usando un pequeño epsilon
    epsilon = 1e-10
    entropia = -np.sum(hist_norm[:, 0] * np.log2(hist_norm[:, 0] + epsilon))

    print(f' Imagen {i + 1}:')
    print(f'  Media:     {media:.2f}')
    print(f'  Varianza:  {varianza:.2f}')
    print(f'  Asimetría: {asimetria:.2f}')
    print(f'  Energía:   {energia:.4f}')
    print(f'  Entropía:  {entropia:.4f}\n')   
    plt.figure(i+10)
    plt.plot(histo, color='black')
    plt.title(f'Histograma Imagen {nombres[i]}')
plt.show()
#LE PEGUE A TODOS 

'''Media: Brillo promedio.

Varianza: Contraste (cuánto se alejan los valores de la media).

Asimetría: Si los tonos tienden hacia oscuros (<0) o claros (>0).

Energía: Medida de uniformidad. Alta cuando hay pocos tonos dominantes.

Entropía: Grado de aleatoriedad. Más alta cuando hay más diversidad de niveles.'''