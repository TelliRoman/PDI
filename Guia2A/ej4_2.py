import numpy as np
import cv2 as cv
from PIL import Image

# Calcular distancia (error absoluto)
def calcular_error(img1, img2):
    return np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
    
# Abrir el .gif con PIL
a7_x = Image.open(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\a7v600-X.gif")
a7_xRuido = Image.open(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\a7v600-X(RImpulsivo).gif")
a7_se = Image.open(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\a7v600-SE.gif")
a7_seRuido = Image.open(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\a7v600-SE(RImpulsivo).gif")
# Convertir el primer frame a array de NumPy (compatible con OpenCV)
a7_x = np.array(a7_x.convert("L"))  # También podés usar "L" para escala de grises
a7_xRuido = np.array(a7_xRuido.convert("L"))  # También podés usar "L" para escala de grises
a7_se = np.array(a7_se.convert("RGB"))  # También podés usar "L" para escala de grises
a7_seRuido = np.array(a7_seRuido.convert("RGB"))  # También podés usar "L" para escala de grises

# Convertir RGB a BGR (porque OpenCV usa BGR)
a7_x = cv.cvtColor(a7_x, cv.COLOR_RGB2BGR)
a7_xRuido = cv.cvtColor(a7_xRuido, cv.COLOR_RGB2BGR)
a7_se = cv.cvtColor(a7_se, cv.COLOR_RGB2BGR)    
a7_seRuido = cv.cvtColor(a7_seRuido, cv.COLOR_RGB2BGR)

diferencia = np.abs(a7_x.astype(np.float32) - a7_se.astype(np.float32))
diferencia = np.where(diferencia > 0, 255, 0).astype(np.uint8)
mask_a7_x = cv.bitwise_and(a7_x,diferencia)
mask_a7_se = cv.bitwise_and(a7_se,diferencia)
mask_a7_seRuido = cv.bitwise_and(a7_seRuido,diferencia)
mask_a7_xRuido = cv.bitwise_and(a7_xRuido,diferencia)

error_x = calcular_error(mask_a7_xRuido, mask_a7_x)
error_se = calcular_error(mask_a7_xRuido, mask_a7_se)

# --- DECISIÓN ---
print(f"Error con modelo X:  {error_x}")
print(f"Error con modelo SE: {error_se}")

if error_x < error_se:
    print("La imagen corresponde a una placa A7V600-X")
else:
    print("La imagen corresponde a una placa A7V600-SE")

# --- OPCIONAL: Mostrar resultados ---
cv.imshow("Imagen Entrada", a7_seRuido)
cv.imshow("Zona Dif",diferencia)
cv.imshow("Modelo X", a7_x)
cv.imshow("Modelo SE", a7_se)

cv.waitKey(0)
cv.destroyAllWindows()


