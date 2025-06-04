import cv2
import os
from feat import Detector

# Inicializar el detector
detector = Detector()

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Guardamos el frame como imagen temporal
    temp_img_path = "frame.jpg"
    cv2.imwrite(temp_img_path, frame)

    # Detectar AUs en la imagen
    try:
        result = detector.detect_image(temp_img_path)
        aus = result.aus
        faces = result.facebox

        # Mostrar AU más activas
        if not aus.empty:
            top_aus = aus.iloc[0].sort_values(ascending=False).head(3)
            y_offset = 30
            for au_name, score in top_aus.items():
                cv2.putText(frame, f"{au_name}: {score:.2f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

        # Dibujar caja sobre el rostro si se detectó
        if not faces.empty:
            x, y, w, h = map(int, faces.iloc[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    except Exception as e:
        print("Error:", e)

    # Mostrar imagen
    cv2.imshow("AU Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
