import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import winsound  # Para la alarma
import time

# Ruta al archivo de sonido convertido a .wav
alarma_path = r"D:\AB\proyecto_videovigilancia\proyecto-env\Conjunta\alarma.wav"

# Cargar los datos de calibración de la cámara desde los archivos XML
def load_calibration_data():
    cv_file = cv2.FileStorage("aruco_calibration/calibration_data.xml", cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("cameraMatrix").mat()
    dist_coeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_calibration_data()

# Obtener la nueva matriz de cámara para la corrección de distorsión
cap = cv2.VideoCapture(1)
imageSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, imageSize, 1, imageSize)

# Cargar el modelo de detección de personas desde TensorFlow Hub
detector_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

# Función para manejar la alarma
def manejar_alarma(activa):
    if activa:
        winsound.PlaySound(alarma_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        winsound.PlaySound(None, winsound.SND_PURGE)

# Detección de personas utilizando TensorFlow en un hilo separado
def detect_persons(frame, results):
    input_tensor = cv2.resize(frame, (320, 320))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    
    detections = detector_model(input_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    results['boxes'] = detection_boxes
    results['scores'] = detection_scores
    results['classes'] = detection_classes

# Detección de marcadores ArUco y delimitación de la zona
def detect_markers(frame, results, last_detection_time, detection_duration=2.5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    current_time = time.time()

    if marker_ids is not None and len(marker_ids) >= 4:
        # Calcular los centros de los marcadores
        centers = []
        for corner in marker_corners:
            center_x = int(np.mean(corner[0][:, 0]))
            center_y = int(np.mean(corner[0][:, 1]))
            centers.append([center_x, center_y])

        pts = np.array(centers, np.int32).reshape((-1, 1, 2))
        results['marker_corners'] = pts
        last_detection_time['time'] = current_time
    else:
        # Mantener la zona visible durante un breve período después de perder los marcadores
        if current_time - last_detection_time['time'] < detection_duration:
            results['marker_corners'] = last_detection_time.get('last_corners')
        else:
            results['marker_corners'] = None

    # Actualizar la última posición conocida de los marcadores
    if results['marker_corners'] is not None:
        last_detection_time['last_corners'] = results['marker_corners']

# Bucle principal de procesamiento de video
last_detection_time = {'time': 0, 'last_corners': None}

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame de la cámara")
        break

    # Aplicar la corrección de distorsión de la cámara
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newCameraMatrix)

    # Recortar la imagen según la ROI obtenida en la corrección de distorsión
    x, y, w, h = roi
    frame = frame[y:y+h, x:x+w]

    results = {'boxes': None, 'scores': None, 'classes': None, 'marker_corners': None}

    # Crear hilos para detección de ArUco y personas
    thread_aruco = threading.Thread(target=detect_markers, args=(frame, results, last_detection_time))
    thread_personas = threading.Thread(target=detect_persons, args=(frame, results))

    thread_aruco.start()
    thread_personas.start()

    thread_aruco.join()
    thread_personas.join()

    person_in_zone = False
    status_message = "Estado: Obra de arte segura"
    intruso_count = 0

    # Procesar resultados de detección de personas y ArUco
    if results['marker_corners'] is not None:
        # Crear una máscara de la zona delimitada por los ArUcos
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [results['marker_corners']], 255)

        # Procesar resultados de detección de personas
        if results['scores'] is not None:
            for i in range(len(results['scores'])):
                if results['scores'][i] > 0.5 and int(results['classes'][i]) == 1:  # Clase 1 es persona
                    box = results['boxes'][i]
                    h, w = frame.shape[:2]
                    y1, x1, y2, x2 = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)

                    intruso_count += 1
                    intruso_label = f"Intruso {intruso_count}"

                    # Dibujar el cuadro alrededor de cada persona detectada en verde
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, intruso_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    # Verificar si cualquier parte de la persona está dentro de la zona delimitada
                    if mask[y1:y2, x1:x2].any():
                        rectangle_color = (0, 0, 255)
                        person_in_zone = True
                        status_message = f"¡Cuidado! intruso detectado"

        # Dibujar el rectángulo alrededor de los ArUcos
        cv2.polylines(frame, [results['marker_corners']], isClosed=True, color=(0, 0, 255) if person_in_zone else (0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    # Manejar la alarma en función de si hay una persona en la zona
    manejar_alarma(person_in_zone)

    # Mostrar el mensaje en la parte superior de la imagen
    color_texto = (0, 255, 0) if not person_in_zone else (0, 0, 255)
    cv2.putText(frame, status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_texto, 3, cv2.LINE_AA)

    # Mostrar el frame con las detecciones
    cv2.imshow('Detección de Personas y Zona Delimitada por ArUcos', frame)

    # Verifica si se presiona 'q' o se cierra la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()