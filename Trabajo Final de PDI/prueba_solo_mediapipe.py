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

# Cálculo de la intensidad de AUs necesarias
def calculate_pain_intensity(face_landmarks, w, h):
    # AU4: Brow Lower (cejas hacia abajo)
    d_au4 = distancia(punto( brow_l:=66, face_landmarks, w, h), punto(brow_r:=296, face_landmarks, w, h))

    # AU6: Cheek Raiser (mejilla sube)
    d_au6 = distancia(punto(159, face_landmarks, w, h), punto(205, face_landmarks, w, h))  # ojo-mejilla izq

    # AU7: Lids Tight (párpado contraído)
    d_au7 = distancia(punto(159, face_landmarks, w, h), punto(145, face_landmarks, w, h))  # párpado superior-inf

    # AU9: Nose Wrinkler (nariz fruncida)
    d_au9 = distancia(punto(6, face_landmarks, w, h), punto(4, face_landmarks, w, h))

    # AU10: Upper Lip Raiser (labio superior sube)
    d_au10 = distancia(punto(13, face_landmarks, w, h), punto(14, face_landmarks, w, h))

    # AU43: Eyes Closed (ojos cerrados)
    d_au43 = (distancia(punto(159, face_landmarks, w, h), punto(145, face_landmarks, w, h)) +
              distancia(punto(386, face_landmarks, w, h), punto(374, face_landmarks, w, h))) / 2

    # Normalización simple: valores bajos = mayor activación muscular
    # Para hacerlo proporcional, usamos una constante arbitraria para escalar (puede calibrarse)
    scale = 50.0  # Cuanto menor sea la distancia, mayor la AU (1.0 - valor normalizado)
    AU4 = 1.0 - min(d_au4 / scale, 1.0)
    AU6 = 1.0 - min(d_au6 / scale, 1.0)
    AU7 = 1.0 - min(d_au7 / scale, 1.0)
    AU9 = 1.0 - min(d_au9 / scale, 1.0)
    AU10 = 1.0 - min(d_au10 / scale, 1.0)
    AU43 = 1.0 - min(d_au43 / scale, 1.0)

    # Fórmula del dolor propuesta
    pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

    return round(pain, 2), {"AU4": AU4, "AU6": AU6, "AU7": AU7, "AU9": AU9, "AU10": AU10, "AU43": AU43}
# Inicialización de los módulos de MediaPipe para dibujo y detección holística
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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
            pain_level, aus = calculate_pain_intensity(results.face_landmarks.landmark, w, h)

            # Mostrar en pantalla
            cv2.putText(annotated, f"Dolor estimado: {pain_level:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # AU mappings según la imagen
            au_landmarks = {
    # AU4: Ceja fruncida (Brow Lowerer) - Puntos clave en ambas cejas
    "AU4": [70, 63, 53, 105, 52, 66, 65, 107, 55, 285, 336, 295, 296, 282, 334, 283, 293, 300],  # Ceja derecha (336=inicio, 293=centro)

    # AU6: Mejilla elevada (Cheek Raiser) - Puntos alrededor de los ojos y mejillas
    "AU6": [352, 346, 347, 280, 266, 330, 425, 118, 101, 142, 36, 50, 117, 123],  # Ojo derecho + mejilla

    # AU7: Párpados tensos (Lid Tightener) - Puntos de párpados superiores e inferiores
    "AU7": [381, 380, 477, 477, 373, 390, 256, 252, 253, 254, 339, 255, 359, 446, 26, 154, 22, 153, 23, 145, 24, 144, 110, 163, 25, 163, 7, 130, 256, 341, 382],  # Párpado derecho

    # AU9: Nariz arrugada (Nose Wrinkler) - Puntos en la nariz y alrededor
    "AU9": [64, 235, 235, 98, 327, 294, 455, 278, 360, 363, 281, 5, 51, 134, 131, 64, 102, 331, 131, 131, 134, 79, 239, 44, 274, 459, 457, 309, 289, 59, 131, 134],  # Nariz (4=punta, 6=puente)

    # AU10: Labio superior elevado (Upper Lip Raiser) - Puntos en el labio superior y nariz
    "AU10": [185, 74, 39, 37, 0, 267, 269, 270, 409, 272, 271, 268, 12, 38, 41, 40],  # 0=base nariz, 13/14=labio

    # AU43: Ojos cerrados (Eyes Closed) - Puntos de párpados (similar a AU7 pero más específico)
    "AU43": [382, 381, 380, 477, 373, 390, 249, 263, 466, 388, 387, 386, 385, 476, 381, 384, 398, 173, 157, 158, 159, 160, 161, 7, 33, 163, 144, 145, 153, 154] # Ojo derecho
}

            au_colors = {
    "AU4": (255, 0, 0),       # Rojo para cejas
    "AU6": (0, 255, 0),       # Verde para mejillas
    "AU7": (0, 128, 255),     # Celeste para párpados
    "AU9": (0, 0, 255),       # Azul para nariz
    "AU10": (255, 0, 255),    # Fucsia para labio superior
    "AU43": (255, 255, 0)     # Amarillo para ojos cerrados
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
