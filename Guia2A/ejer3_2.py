import cvui
import numpy as np
import cv2
def promediado(x, frames):
    suma = np.zeros_like(frames[0], dtype=np.float32)  # Inicializamos suma con ceros del tamaño de una imagen
    for i in range(0, x):
        suma += frames[i].astype(np.float32)  # Convertimos los frames a float para evitar desbordamientos
    promedio = (suma / x).astype(np.uint8)  # Escalamos de vuelta a valores de 8 bits
    cv2.imshow("Cantidad", promedio)

cap=cv2.VideoCapture(r"Imagenes\pedestrians.mp4")

frames=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    if (not ret):
        break 
    else:
        frames.append(frame)

def trackbar_callback(x):
    if x > 0:
        promediado(x, frames)

cv2.namedWindow("Cantidad")
cv2.createTrackbar("Frames", "Cantidad", 0, len(frames), trackbar_callback)
promediado(1, frames)  # Mostrar el promedio inicial (sin frames seleccionados)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


    