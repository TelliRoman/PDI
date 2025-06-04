import openface
import cv2

# Cargar el modelo de red neuronal
net = openface.TorchNeuralNet('models/openface/nn4.small2.v1.t7', 96)

# Capturar video desde la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar la cara más grande en la imagen
    bb = openface.AlignDlib('models/openface/shape_predictor_68_face_landmarks.dat').getLargestFaceBoundingBox(frame)
    if bb is not None:
        aligned_face = openface.AlignDlib('models/openface/shape_predictor_68_face_landmarks.dat').align(96, frame, bb)
        rep = net.forward(aligned_face)
        # Aquí puedes procesar 'rep' para obtener las AUs y otros análisis

    # Mostrar el frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
