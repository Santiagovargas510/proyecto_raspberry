import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

# Cargar modelo YOLO para detectar personas
model = YOLOWorld(model_id="yolo_world/l")
classes = ["person"]  # Detecta solo personas
model.set_classes(classes)

# Configuración de anotadores
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)

# Función simulada para la clasificación de género
def classify_gender(cropped_person):
    import random
    return "Male" if random.random() > 0.5 else "Female"

# Iniciar captura de video
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia de detección de personas
    results = model.infer(frame, confidence=0.4)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.4)

    # Anotación de detecciones y clasificación de género
    for detection in detections:
        # Extraer las coordenadas desde el primer elemento del tuple
        x1, y1, x2, y2 = map(int, detection[0])  # detection[0] contiene las coordenadas

        cropped_person = frame[y1:y2, x1:x2]  # Extrae la región de la persona detectada

        # Clasificación de género
        gender_label = classify_gender(cropped_person)
      

        # Actualizar el label de la detección para mostrar solo el género
        detection[5]["class_name"] = f"{gender_label}"  # Cambia la etiqueta para mostrar solo el género

    # Anotación de detecciones
    annotated_frame = BOX_ANNOTATOR.annotate(frame, detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)

    cv2.imshow("Detección de género", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
