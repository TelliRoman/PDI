import numpy as np
import cv2
import mediapipe as mp
def calcular_angulo_rotacion(punto_centroide, punto_izquierdo, punto_derecho, vector_referencia=np.array([0,0,-1])):
    """
    Calcula el ángulo entre la normal definida por los puntos
    (punto_izquierdo, punto_derecho, punto_centroide) y un vector de referencia.

    Parámetros:
        punto_centroide: array-like de 3 elementos (x,y,z)
        punto_izquierdo: array-like de 3 elementos (x,y,z)
        punto_derecho: array-like de 3 elementos (x,y,z)
        vector_referencia: array-like de 3 elementos, vector para comparar (default [0,0,1])

    Retorna:
        angulo (float)
    """
    # Convertir a numpy arrays
    centroide = np.array(punto_centroide, dtype=float)
    izquierdo = np.array(punto_izquierdo, dtype=float)
    derecho = np.array(punto_derecho, dtype=float)
    ref = np.array(vector_referencia, dtype=float)

    # Vectores desde centroide a los puntos de los ojos
    v1 = izquierdo - centroide
    v2 = derecho - centroide

    # Vector normal al plano definido por esos dos vectores (producto cruzado)
    normal = np.cross(v1, v2)
    norma = np.linalg.norm(normal)

    if norma == 0:
        raise ValueError("Los puntos están alineados, no se puede calcular normal.")

    normal_u = normal / norma

    # Normalizar vector de referencia
    ref_u = ref / np.linalg.norm(ref)

    # Calcular el ángulo con clip para evitar errores numéricos
    dot = np.clip(np.dot(normal_u, ref_u), -1.0, 1.0)
    angulo =np.degrees(np.arccos(dot))

    return angulo
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
            #Puntos para el centroide
            cien_derecha=face_landmarks[368]
            cien_izquierda=face_landmarks[139]
            centroide_x = int(((cien_izquierda.x*w) + (cien_derecha.x*w) )/ 2)
            centroide_y = int(((cien_izquierda.y*h) + (cien_derecha.y*h) )/ 2)
            centroide_z = int(((cien_izquierda.z*h) + (cien_derecha.z*h) )/ 2)
            
            cien_izquierda = np.array([face_landmarks[139].x, face_landmarks[139].y, face_landmarks[139].z])
            cien_derecha = np.array([face_landmarks[368].x, face_landmarks[368].y, face_landmarks[368].z])

            angulo=calcular_angulo_rotacion(cien_derecha,cien_izquierda,(centroide_x,centroide_y,centroide_z))
            if angulo > 50:
                print("MIRA A LA CAMARA SORETE")
            # AU mappings según la imagen
            au_landmarks = {
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
            
            cv2.circle(annotated, (centroide_x,centroide_y), 2, (255,255,255), -1)

        cv2.imshow("MediaPipe + Calidad de Imagen", annotated)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberamos la cámara y cerramos ventanas
cap.release()
cv2.destroyAllWindows()
