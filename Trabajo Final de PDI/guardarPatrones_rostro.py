import numpy as np
import cv2
import time
import mediapipe as mp
from normalizarKeypoints_Rostro import *

# Inicialización de los módulos de MediaPipe para dibujo y detección holística
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Inicializamos la cámara
cap = cv2.VideoCapture(0)
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
frames_a_guardar = 500  # Número de frames a guardar
# Inicializar tiempo
prev_frame_time = 0
             
# Inicializar variables para saber si ya se cargaron los datos de postura
datos_de_no_dolor = 0
datos_dolor = 0
guardando_no_dolor = False
guardando_dolor = False   
 



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
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = current_time

        # Voltear horizontalmente para espejo
        frame = cv2.flip(frame, 1)
        
        # Convertimos a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesamos el frame con MediaPipe Holistic
        results = holistic.process(rgb_frame)
        
        h, w, _ = frame.shape
        # Copiamos el frame para dibujar sobre él
        annotated = frame.copy()
        
        # --- DIBUJO DE LOS PUNTOS Y CONEXIONES ---
        if results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
        
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
            
        # Si se presiona 'c', cambiar el estado para pasar a guardar datos de postura
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if datos_de_no_dolor == 0: # Si no ha
                guardando_no_dolor = True #
            elif datos_dolor == 0 and not guardando_dolor: # Si se finalizó el guardado de buena postura y no hay datos de mala postura
                guardando_dolor = True # 
          # Guardar keypoints de buena postura
        if guardando_no_dolor and results.face_landmarks:
            if datos_de_no_dolor < frames_a_guardar:
                keypoints = []
                for au, indices in au_landmarks.items():
                    for idx in indices:
                            lm = face_landmarks[idx]
                            keypoints.extend([lm.x, lm.y, lm.z])
                norm_keypoints = normalizarKeypoints(keypoints)   
                with open(f'Trabajo Final de PDI/datos_dolor/no_dolor{datos_de_no_dolor}.csv', "w") as f:
                    f.write(",".join(map(str, norm_keypoints)))
                datos_de_no_dolor += 1
            else:
                guardando_no_dolor = False
        
        if guardando_dolor and results.face_landmarks:
            if datos_dolor < frames_a_guardar:
                keypoints = []
                for au, indices in au_landmarks.items():
                    for idx in indices:
                            lm = face_landmarks[idx]
                            keypoints.extend([lm.x, lm.y, lm.z])
                norm_keypoints = normalizarKeypoints(keypoints)   
                with open(f'Trabajo Final de PDI/datos_dolor/dolor{datos_dolor}.csv', "w") as f:
                    f.write(",".join(map(str, norm_keypoints)))
                datos_dolor += 1
            else:
                guardando_dolor= False

            # Mostrar distintos mensajes según si se han cargado los datos de postura
        if datos_de_no_dolor == 0:
            text = "Presiona 'C' para guardar datos de rostro de no dolor"
            color = (255, 255, 255)  # Blanco
        elif datos_dolor == 0:
            text = "Presiona 'C' para guardar datos de dolor"
            color = (255, 255, 255)  # Blanco
        else:
            text = "Datos de rostro guardados correctamente"
            color = (255, 255, 255)  # Blanco

        if guardando_no_dolor:
            text = "Guardando datos de no dolor..."
            color = (255, 255, 255)  # Blanco

        if guardando_dolor:
            text = "Guardando datos de dolor..."
            color = (255, 255, 255)  # Blanco
            # Mostrar el texto en la imagen centrado abajo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)


        # Calcular coordenadas para centrar abajo
        x = (w - text_width) // 2
        y = h - 20  # 20 píxeles de margen inferior
        cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Mostrar FPS en esquina superior izquierda
        fps_text = f"FPS: {int(fps)}"
        fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
        cv2.putText(annotated, fps_text, (10, fps_y+10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Mostrar cantidad de buena postura en esquina superior izquierda
        fps_text = f"No dolor: {int(datos_de_no_dolor)}"
        fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
        cv2.putText(annotated, fps_text, (10, fps_y+25), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Mostrar cantidad de buena postura en esquina superior izquierda
        fps_text = f"Dolor: {int(datos_dolor)}"
        fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
        cv2.putText(annotated, fps_text, (10, fps_y+40), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrar el frame con puntos
        cv2.imshow("Posture Keypoints", annotated)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
# Liberamos la cámara y cerramos ventanas
cap.release()
cv2.destroyAllWindows()

            