import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import winsound  # Para la alarma (si lo usas en Windows)

# Cargar los datos de calibración de la cámara desde los archivos XML
def load_calibration_data():
    cv_file = cv2.FileStorage("aruco_calibration/calibration_data.xml", cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("cameraMatrix").mat()
    dist_coeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_calibration_data()

# Obtener la nueva matriz de cámara para la corrección de distorsión
def process_frame(frame, camera_matrix, dist_coeffs, newCameraMatrix):
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newCameraMatrix)
    return frame

# Función principal
def main():
    st.title("Detección de Personas y Zona Delimitada por ArUcos")

    # Cargar el modelo de detección de personas desde TensorFlow Hub
    detector_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

    # Iniciar la captura de video
    cap = cv2.VideoCapture(1)
    imageSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, imageSize, 1, imageSize)

    last_detection_time = {'time': 0, 'last_corners': None}

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo capturar el frame de la cámara")
            break

        frame = process_frame(frame, camera_matrix, dist_coeffs, newCameraMatrix)
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]

        results = {'boxes': None, 'scores': None, 'classes': None, 'marker_corners': None}

        # Aquí iría el código para detección de personas y ArUco

        st.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
