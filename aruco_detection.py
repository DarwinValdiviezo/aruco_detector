import cv2
import numpy as np
import xml.etree.ElementTree as ET

def load_calibration_data():
    cv_file = cv2.FileStorage("aruco_calibration/calibration_data.xml", cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("cameraMatrix").mat()
    dist_coeffs = cv_file.getNode("distCoeffs").mat()
    cv_file.release()
    return camera_matrix, dist_coeffs

def load_aruco_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    markers = []
    for elem in root.findall('marker_42'):
        center = elem.text.strip().split()
        center = tuple(map(float, center))
        markers.append(center)
    
    return markers

def detect_custom_aruco_markers(frame, marker_centers):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral y encontrar contornos
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_markers = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Considerar contornos con 4 vértices
            (x, y, w, h) = cv2.boundingRect(approx)
            if w > 30 and h > 30:  # Filtrar contornos pequeños
                marker_center = (x + w / 2, y + h / 2)
                valid_markers.append(marker_center)

    return valid_markers

def draw_marker_rectangle(frame, corners):
    if len(corners) == 4:
        marker_centers = sorted(corners, key=lambda x: (x[0], x[1]))

        for i in range(4):
            next_i = (i + 1) % 4
            pt1 = (int(marker_centers[i][0]), int(marker_centers[i][1]))
            pt2 = (int(marker_centers[next_i][0]), int(marker_centers[next_i][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    return frame
