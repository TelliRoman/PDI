import numpy as np
import cv2
import mediapipe as mp
import pickle
from normalizarKeypoints_Rostro import *
from pre_processing_functions import is_blurry, get_brightness, sharpen_image, adjust_gamma



def calcular_angulo_rotacion(punto_centroide, punto_izquierdo, punto_derecho, vector_referencia=np.array([0,0,-1])):
    """Calcula el ángulo entre la normal definida por los puntos
    (punto_izquierdo, punto_derecho, punto_centroide) y un vector de referencia.
    Parámetros:
        punto_centroide: array-like de 3 elementos (x,y,z)
        punto_izquierdo: array-like de 3 elementos (x,y,z)
        punto_derecho: array-like de 3 elementos (x,y,z)
        vector_referencia: array-like de 3 elementos, vector para comparar (default [0,0,1])
    Retorna:
        angulo (float)"""
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
# AU mappings según la imagen
au_landmarks = {
    "AU4": [70, 63, 53, 105, 52, 66, 65, 107, 55, 285, 336, 295, 296, 282, 334, 283, 293, 300],
    "AU6": [352, 346, 347, 280, 266, 330, 425, 118, 101, 142, 36, 50, 117, 123],
    "AU7": [381, 380, 477, 373, 390, 256, 252, 253, 254, 339, 255, 359, 446, 26, 154, 22, 153, 23 ,145, 24, 144, 110, 163, 25, 7, 130, 341, 382],
    "AU9": [64, 235, 98, 327, 294, 455, 278, 360, 363, 281, 5, 51, 134, 131, 102, 331, 79, 239,44, 274, 459, 457, 309, 289, 59],
    "AU10": [185, 74, 39, 37, 0, 267, 269, 270, 409, 272, 271, 268, 12, 38, 41, 40],
    "AU43": [382, 380, 477, 373, 390, 249, 263, 466, 388, 387, 386, 385, 476, 384, 398, 173, 157, 158, 159, 160, 161, 7, 33, 163, 144, 145, 153, 154,314, 405, 321, 17, 84, 181, 91, 146, 320, 403, 316, 15, 86, 179, 90, 180, 85, 315, 16, 404, 320],
    "AUDEMAS":[291, 61, 57, 186, 165, 92, 167, 167, 167, 164, 393, 391, 322, 410, 8, 9, 168, 6, 197, 195, 5, 281, 363, 360, 51, 134, 131, 49, 5, 3, 236, 248, 456, 419, 196, 351, 122],
    "AUCIEN": [368, 139]
}
au_colors = {
            "AU4": (255, 0, 0),       # Rojo para cejas
            "AU6": (0, 255, 0),       # Verde para mejillas
            "AU7": (0, 128, 255),     # Celeste para párpados
            "AU9": (0, 0, 255),       # Azul para nariz
            "AU10": (255, 0, 255),    # Fucsia para labio superior
            "AU43": (255, 255, 0) ,    # Amarillo para ojos cerrados
            "AUDEMAS": (0, 255, 255),   # Naranja para cejas y ojos
            "AUCIEN": (255, 255, 255)  # Blanco para AUCIEN
            }

# Cargar modelo entrenado
with open("Trabajo Final de PDI/modelo_dolor.pkl", "rb") as f:
    model = pickle.load(f)
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
         # Voltear horizontalmente para espejo
        frame = cv2.flip(frame, 1)
        # Convertimos a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesamos el frame con MediaPipe Holistic
        results = holistic.process(rgb_frame)
        # Copiamos el frame para dibujar sobre él
        annotated = frame.copy()
       
        h, w, _ = frame.shape
        # --- DIBUJO DE LOS PUNTOS Y CONEXIONES ---
        if results.face_landmarks:
           
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
            if angulo > 40:
                label= "Porfavor Mira hacia la camara"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                x = (w - text_width) // 2
                y = h - 20  # 20 píxeles de margen inferior
                cv2.putText(annotated, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            else:
                #PREPROCESAMIENTO
                # Verificar si la imagen está borrosa
                blurry, blur_score = is_blurry(frame)
                if blurry:
                    frame = sharpen_image(frame)
                brightness = get_brightness(frame)
                illumination_state = (
                        "Oscura" if brightness < 60 else
                        "Normal" if brightness < 200 else
                        "Sobreexpuesta"
                )
                if illumination_state == "Oscura":
                    frame = adjust_gamma(frame, gamma=1.5)
                elif illumination_state == "Sobreexpuesta":
                    frame = adjust_gamma(frame, gamma=0.7)
                #NORMALIZACION DE KEYPOINTS
                keypoints = []
                for au, indices in au_landmarks.items():
                   for idx in indices:
                        lm = face_landmarks[idx]
                        keypoints.extend([lm.x, lm.y, lm.z])
                norm_keypoints = normalizarKeypoints(keypoints)
                #PREDICCION DE DOLOR
                # Clasificar dolor
                prediction = model.predict(np.array([norm_keypoints]))[0]
                label = "No dolor" if prediction == "no_dolor" else "Dolor"
                color = (0, 255, 0) if prediction == "no_dolor" else (0, 0, 255)

                # Mostrar resultado centrado abajo
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x = (w - text_width) // 2
                y = h - 20  # 20 píxeles de margen inferior
                cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            

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
