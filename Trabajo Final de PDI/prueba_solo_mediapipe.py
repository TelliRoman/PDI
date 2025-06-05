import numpy as np
import cv2
import mediapipe as mp

# Distancia euclídea entre dos puntos
def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Función para convertir índice a punto en píxeles
def punto(idx, face_landmarks, w, h):
    lm = face_landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

# Alinear rostro con transformación afín
def alinear_rostro(frame, face_landmarks, w, h):
    ojo_izq = punto(33, face_landmarks, w, h)
    ojo_der = punto(263, face_landmarks, w, h)
    centro = punto(1, face_landmarks, w, h)  # entrecejo
    #Tamaño deseado del rostro alineado.
    ancho = 200
    alto = 200
    #Define las posiciones objetivo (en la imagen alineada) para los 3 puntos clave:
    destino = np.float32([[60, 100], [140, 100], [100, 140]])
    origen = np.float32([ojo_izq, ojo_der, centro])
    #Calcula la matriz de transformación afín M para llevar los puntos origen a las posiciones destino.
    M = cv2.getAffineTransform(origen, destino)
    #Aplica la transformación M a toda la imagen, alineando el rostro.
    frame_alineado = cv2.warpAffine(frame, M, (w, h))
    #Devuelve la imagen alineada y la matriz de transformación M.
    return frame_alineado, M

# Aplicar transformación afín a un punto
def transformar_punto(pt, M):
    p = np.array([pt[0], pt[1], 1])
    #Multiplica la matriz M por el vector p para obtener el punto transformado.
    res = M @ p
    return (res[0], res[1])

# Cálculo del dolor con normalización
def calculate_pain_intensity_normalized(face_landmarks, w, h, M):
    #transforma el punto idx usando la matriz M.
    def t(idx): return transformar_punto(punto(idx, face_landmarks, w, h), M)

    # Referencia: distancia entre ojos (escalado adaptativo)
    ref_dist = distancia(t(33), t(263))
    if ref_dist < 1.0: ref_dist = 1.0  # evitar división por cero

    # Medidas
    d_au4 = distancia(t(66), t(296))       # Cejas
    d_au6 = distancia(t(159), t(205))      # Mejillas / contracción orbicular
    d_au7 = distancia(t(159), t(145))      # Elevación párpado inferior
    d_au9 = distancia(t(6), t(4))          # Nariz
    d_au10 = distancia(t(13), t(14))       # Boca
    d_au43 = (distancia(t(159), t(145)) + distancia(t(386), t(374))) / 2  # Cierre ojos

    # Normalización relativa al ancho facial (ref_dist) Cada medida se normaliza dividiendo por una fracción de ref_dist y restando de 1.0. 
    AU4 = 1.0 - min(d_au4 / (ref_dist * 0.45), 1.0)
    AU6 = 1.0 - min(d_au6 / (ref_dist * 0.30), 1.0)
    AU7 = 1.0 - min(d_au7 / (ref_dist * 0.20), 1.0)
    AU9 = 1.0 - min(d_au9 / (ref_dist * 0.15), 1.0)
    AU10 = 1.0 - min(d_au10 / (ref_dist * 0.15), 1.0)
    AU43 = 1.0 - min(d_au43 / (ref_dist * 0.18), 1.0)
    #Estos valores estiman la activación de unidades de acción (Action Units) de FACS (Facial Action Coding System).
    pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

    return round(pain, 2), {"AU4": AU4, "AU6": AU6, "AU7": AU7, "AU9": AU9, "AU10": AU10, "AU43": AU43}

# Función para determinar si una imagen está borrosa usando la varianza del Laplaciano
def is_blurry(frame, threshold=100.0):
    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculamos la varianza del Laplaciano (detecta cambios de bordes)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Si la varianza es menor al umbral, se considera borrosa
    return lap_var < threshold, lap_var

# Función para medir la iluminación usando el canal V (brillo) del espacio HSV
def get_brightness(frame):
    # Convertimos a HSV para trabajar con el brillo
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Canal V
    return np.mean(v_channel)  # Promedio del brillo

# Función para aplicar un desenfoque suave (máscara difusa)
def apply_soft_mask(frame, ksize=(5, 5)):
    # Aplicamos un filtro Gaussiano, útil para estabilizar ruido o movimientos bruscos
    return cv2.GaussianBlur(frame, ksize, 0)

#Filtro de acentuado 
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Corrección gamma para mejorar brillo u oscurecer
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Normalización del canal V (cuando el rango dinámico es bajo)
def normalize_v_channel(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v = 255 * (v - v.min()) / (v.max() - v.min() + 1e-6)
    hsv[:, :, 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Inicialización de los módulos de MediaPipe para dibujo y detección holística
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# Inicializamos la cámara
cap = cv2.VideoCapture(0)

# Inicializamos MediaPipe Holistic (rostro + cuerpo + manos)
with mp_holistic.Holistic(
    static_image_mode=False,           # True si analizamos imágenes estáticas
    model_complexity=1,                # 0 = rápido pero menos preciso, 1 = balanceado, 2 = más preciso pero más lento
    smooth_landmarks=True,             # Suaviza las detecciones en video
    enable_segmentation=False,         # No segmentamos el cuerpo
    refine_face_landmarks=True,        # Detección refinada de rostro
    min_detection_confidence=0.5,      # Confianza mínima para detectar
    min_tracking_confidence=0.5        # Confianza mínima para seguir puntos
) as holistic:
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Evaluamos borrosidad (Laplace)
        blurry, blur_score = is_blurry(frame)
        if blurry:
            # Si es borrosa se puede aplicar filtro de acentuado o continuar
            #continue
            frame = sharpen_image(frame)

        # Medimos el brillo de la escena
        brightness = get_brightness(frame)

        # Interpretamos el nivel de iluminación
        illumination_state = (
            "Oscura" if brightness < 60 else
            "Normal" if brightness < 200 else
            "Sobreexpuesta"
        )

        # Mejora de imagen según iluminación
        if illumination_state == "Oscura":
            frame = adjust_gamma(frame, gamma=1.5)  # Aclarar
        elif illumination_state == "Sobreexpuesta":
            frame = adjust_gamma(frame, gamma=0.7)  # Oscurecer
        else:
            # Medimos rango dinámico del canal V
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            rango = np.max(v) - np.min(v)
            if rango < 60:
                frame = normalize_v_channel(frame)  # Mejorar contraste general

        # Convertimos a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el frame con MediaPipe Holistic
        results = holistic.process(rgb_frame)

        # Copiamos el frame para dibujar sobre él
        annotated = frame.copy()

        # --- DIBUJO DE LOS PUNTOS Y CONEXIONES ---
        if results.face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.face_landmarks.landmark
            frame_alineado, M = alinear_rostro(frame,face_landmarks , w, h)
            pain_level, aus = calculate_pain_intensity_normalized(face_landmarks, w, h, M)

            # Mostrar en pantalla
            cv2.putText(annotated, f"Dolor estimado: {pain_level:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # AU mappings según la imagen
            au_landmarks = {
                "AU1": [66, 105, 107, 55, 65, 52, 285, 295, 282, 283, 336, 334],
                "AU6": [50, 101, 205, 111, 120, 121, 280, 346, 425, 345, 352, 351],
                "AU9": [6, 197, 195, 5, 4, 275, 294],
                "AU15": [61, 146, 91, 181, 84, 17, 314, 291, 375, 321, 308, 324],
                "AU17": [17, 84, 14, 87, 178, 88, 95],
                "AU44": [159, 145, 153, 154, 386, 374, 382, 385]
            }

            au_colors = {
                "AU1": (255, 0, 0),
                "AU6": (0, 255, 0),
                "AU9": (0, 0, 255),
                "AU15": (255, 255, 0),
                "AU17": (255, 0, 255),
                "AU44": (0, 255, 255)
            }

            # Dibujo de puntos clave por AU
            for au, indices in au_landmarks.items():
                color = au_colors[au]
                for idx in indices:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    cv2.circle(annotated, (x, y), 2, color, -1)

                # Etiqueta del AU
                fx = int(face_landmarks[indices[0]].x * w)
                fy = int(face_landmarks[indices[0]].y * h) - 5
                cv2.putText(annotated, au, (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # --- TEXTO INFORMATIVO EN PANTALLA ---

        # Texto sobre borrosidad y valor de varianza
        cv2.putText(annotated, f"Borroso: {'SI' if blurry else 'NO'} | Var: {blur_score:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if blurry else (0, 255, 0), 2)

        # Texto sobre iluminación
        cv2.putText(annotated, f"Luz: {brightness:.1f} ({illumination_state})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mostramos el resultado con todos los datos y dibujos
        cv2.imshow("MediaPipe + Calidad de Imagen", annotated)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberamos la cámara y cerramos ventanas
cap.release()
cv2.destroyAllWindows()
