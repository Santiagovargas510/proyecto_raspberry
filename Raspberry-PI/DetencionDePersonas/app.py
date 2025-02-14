import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import psycopg2
from datetime import datetime, timedelta
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import os

# Conexión a la base de datos PostgreSQL
try:
    conexion = psycopg2.connect(
        host="10.1.0.65",
        user="postgres",
        database="postgres",
        password="o6CG1DBS3a3uxjhYWMuL",
        port=5432
    )
    print("Conexión exitosa a la base de datos")
except psycopg2.OperationalError as e:
    print(f"Error al conectar a la base de datos: {e}")
    exit()

# Modelo YOLO configurado para detectar personas
model = YOLOWorld(model_id="yolo_world/l")
classes = ["person"]
model.set_classes(classes)

# Configuración
DETECTION_TIMEOUT = timedelta(seconds=20)  # Tiempo entre registros
MIN_DISTANCE = 1370  # Distancia máxima para considerar una nueva detección
SIMILARITY_THRESHOLD = 0.4  # Umbral de similitud para considerar misma persona
EXCLUSION_SIMILARITY_THRESHOLD = 0.6  # Umbral para exclusión por similitud

# Lista de rostros detectados recientemente
detected_faces = []  # Formato: [(embedding, timestamp, center)]

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia del modelo YOLO
    results = model.infer(frame, confidence=0.4)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.4)

    current_time = datetime.now()
 
    for detection in detections:
        bbox = detection[0]  # Coordenadas de la caja delimitadora
        x_min, y_min, x_max, y_max = map(int, bbox)
        confidence = float(detection[2])

        # Validar límites del ROI
        height, width, _ = frame.shape
        y_min, y_max = max(0, y_min), min(height, y_max)
        x_min, x_max = max(0, x_min), min(width, x_max)

        # Centro de la detección
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        center_point = np.array([center_x, center_y])
        person_roi = frame[y_min:y_max, x_min:x_max]

        # Validar si el ROI es válido
        if person_roi.size == 0 or person_roi.shape[0] < 20 or person_roi.shape[1] < 20:
            continue

        try:
            face_embedding = DeepFace.represent(person_roi, model_name="Facenet", enforce_detection=False)[0]['embedding']
            # **Detección de género con DeepFace**
            analysis = DeepFace.analyze(person_roi, actions=["gender"], enforce_detection=False)
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            gender = analysis.get("dominant_gender", "NN")
        except Exception as e:
            print(f"Error en la detección facial: {e}")
            gender = "NN"
            continue

        # **Evitar registros duplicados**
        is_duplicate = False
        for prev_embedding, prev_time, prev_center in detected_faces:
            time_diff = (current_time - prev_time).total_seconds()
            distance = np.linalg.norm(center_point - prev_center)
            similarity = 1 - cosine(face_embedding, prev_embedding)

            if time_diff < DETECTION_TIMEOUT.total_seconds() and distance < MIN_DISTANCE and similarity > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # **Guardar el rostro detectado**
        detected_faces.append((face_embedding, current_time, center_point))

        # **Registro en la base de datos**
        valores = (
            "person",
            confidence,
            x_min, y_min, x_max, y_max,
            current_time,
            gender,  # Género detectado
            "zona_1"  # Suponiendo que solo hay una zona por ahora
        )

        query = """
        INSERT INTO contador (clase, confianza, x_min, y_min, x_max, y_max, hora, genero, zona)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor = conexion.cursor()
            cursor.execute(query, valores)
            conexion.commit()
            print("Registro exitoso:", valores)
        except psycopg2.Error as e:
            print(f"Error al insertar en la base de datos: {e}")
            conexion.rollback()

    # Mostrar el video con las detecciones
    cv2.imshow("Detección de personas", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
conexion.close()


#   =======================================================================================

#   import cv2
#   import supervision as sv
#   from inference.models.yolo_world.yolo_world import YOLOWorld
#   import psycopg2
#   from datetime import datetime, timedelta
#   from deepface import DeepFace
#   import numpy as np
#   import time
#   from scipy.spatial.distance import cosine
#   import os

#     Conexión a la base de datos PostgreSQL
#   try:
#       conexion = psycopg2.connect(
#           host="localhost",
#           user="postgres",
#           database="postgres",
#           password="admin",
#           port=5432
#       )
#       print("Conexión exitosa a la base de datos")
#   except psycopg2.OperationalError as e:
#       print(f"Error al conectar a la base de datos: {e}")
#       exit()

#     Modelo YOLO configurado para detectar personas
#   model = YOLOWorld(model_id="yolo_world/l")
#   classes = ["person"]
#   model.set_classes(["person"])


#     Configuración
#   DETECTION_TIMEOUT = timedelta(seconds=20)    Tiempo entre registros
#   REENTRY_TIMEOUT = timedelta(seconds=60)      Tiempo mínimo para reingreso
#   MIN_DISTANCE = 1370    Distancia máxima para considerar una nueva detección
#   SIMILARITY_THRESHOLD = 0.4    Umbral de similitud para considerar misma persona
#   EXCLUSION_SIMILARITY_THRESHOLD = 0.6    Umbral para exclusión por similitud
#   alpha = 1.5    Contraste
#   beta = 30      Brillo

#     Lista de rostros detectados
#     Formato: [(embedding, last_seen, center, exit_time)]
#   detected_faces = []

#     Cargar imágenes de personas excluidas y generar sus embeddings
#   excluded_persons_images = [
#       "imagenes/imagenjuan.jpg",
#       "imagenes/imagensantiago.jpg"
#   ]

#   excluded_faces = []
#   for image_path in excluded_persons_images:
#       try:
#            if not os.path.exists(image_path):
#                print(f"Advertencia: La imagen no existe en la ruta {image_path}")
#                continue
#            img = cv2.imread(image_path)
#            if img is not None:
#                embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)[0]['embedding']
#                excluded_faces.append(embedding)
#                print(f"Embedding generado para: {image_path}")
#       except Exception as e:
#           print(f"Error al procesar la imagen {image_path}: {e}")

#     Zona de interés (coordenadas del cuadro verde)
#   zone_x_min, zone_y_min = 50, 50
#   zone_x_max, zone_y_max = 400, 400

#     Anotador
#   bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color=sv.Color(255, 0, 0))

#     Captura de video
#   cap = cv2.VideoCapture(0)

#   while True:
#       ret, frame = cap.read()
#       if not ret:
#           break

#         Inferencia del modelo YOLO
#       results = model.infer(frame, confidence=0.4)
#       detections = sv.Detections.from_inference(results).with_nms(threshold=0.4)
#       if len(detections) == 0:
#           continue

#       current_time = datetime.now()
 
#       for detection in detections:
#           bbox = detection[0]    Coordenadas de la caja delimitadora
#           x_min, y_min, x_max, y_max = map(int, bbox)
#           confidence = float(detection[2])

#             Validar límites del ROI
#           height, width, _ = frame.shape
#           y_min, y_max = max(0, y_min), min(height, y_max)
#           x_min, x_max = max(0, x_min), min(width, x_max)

#             Centro de la detección
#           center_x = (x_min + x_max) / 2
#           center_y = (y_min + y_max) / 2
#           center_point = np.array([center_x, center_y])
#           person_roi = frame[y_min:y_max, x_min:x_max]

#             Validar si está dentro de la zona de interés
#           if not (zone_x_min <= center_x <= zone_x_max and zone_y_min <= center_y <= zone_y_max):
#               continue

#             Extraer la región de interés (rostro)
#           if person_roi.size == 0 or person_roi.shape[0] < 20 or person_roi.shape[1] < 20:
#               continue

#           try:
#               face_embedding = DeepFace.represent(person_roi, model_name="Facenet", enforce_detection=False)[0]['embedding']
#           except Exception as e:
#               continue

#             Comparar con las huellas excluidas
#           is_excluded = False
#           for excluded_embedding in excluded_faces:
#               score = 1 - cosine(face_embedding, excluded_embedding)
#           if score >= EXCLUSION_SIMILARITY_THRESHOLD:
#               is_excluded = True
#               break

#           if is_excluded:
#               print("Persona excluida detectada, no se registrará.")
#               continue

#             Comparar con embeddings previos y manejar reingresos
#           matched = False
#           for i, (stored_embedding, last_seen, last_center, exit_time) in enumerate(detected_faces):
#               similarity = 1 - cosine(face_embedding, stored_embedding)
#               distance = np.linalg.norm(center_point - last_center)
#               time_since_exit = current_time - exit_time if exit_time else timedelta.max

#               if similarity >= SIMILARITY_THRESHOLD and distance < MIN_DISTANCE:
#                   if time_since_exit < REENTRY_TIMEOUT:
#                       matched = True
#                       break

#                   detected_faces[i] = (stored_embedding, current_time, center_point, None)
#                   matched = True
#                   break

#           if matched:
#               continue

#             Registrar nueva detección
#           detected_faces.append((face_embedding, current_time, center_point, None))
#           print("Nuevo rostro detectado, registrando...")
#           try:
#               analysis = DeepFace.analyze(person_roi, actions=['gender'], enforce_detection=False)
#               gender = analysis[0]['dominant_gender'] if isinstance(analysis, list) else analysis['dominant_gender']
#               time.sleep(1)
#           except Exception as e:
#               gender = "NN"

#           valores = (
#               "person",
#               confidence,
#               x_min, y_min, x_max, y_max,
#               current_time,
#               gender
#           )

#           query = """
#           INSERT INTO contador (clase, confianza, x_min, y_min, x_max, y_max, hora, genero)
#           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#           """
#           try:
#               cursor = conexion.cursor()
#               cursor.execute(query, valores)
#               print("Registro exitoso:", valores)
#               conexion.commit()
#           except psycopg2.Error as e:
#               conexion.rollback()

#       cv2.imshow("Detección de personas", frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break

#   cap.release()
#   cv2.destroyAllWindows()
#   conexion.close()
