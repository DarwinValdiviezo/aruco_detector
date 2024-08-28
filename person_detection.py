import cv2
import tensorflow as tf
import numpy as np

def load_yolo_model():
    # Cargar un modelo de TensorFlow Hub más ligero si es necesario o usar un modelo preentrenado más rápido
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def detect_person(frame, model):
    height, width = frame.shape[:2]

    # Preprocesar la imagen para el modelo MobileNetV2
    image_resized = cv2.resize(frame, (224, 224))
    image_normalized = image_resized / 255.0
    image_reshaped = np.reshape(image_normalized, (1, 224, 224, 3))

    # Realizar la detección
    predictions = model.predict(image_reshaped)

    person_boxes = []

    # Analizar las predicciones
    for pred in predictions:
        class_id = np.argmax(pred)
        if class_id == 15:  # Clase 15 es "persona" en MobileNetV2
            # Puedes ajustar estas posiciones según la salida real del modelo
            x, y, w, h = 0, 0, width, height  # Esto es solo un placeholder
            person_boxes.append([x, y, w, h])
            label = "Intruso"
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, person_boxes
