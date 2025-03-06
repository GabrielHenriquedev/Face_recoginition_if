import pickle
import uuid
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from views.LoadingWindow import ProgressBarWindow

# Novas importações
import platform
import subprocess
from plyer import notification  # Para notificações do sistema
import winsound  #

from views.notification import NotificationWindow


class FaceRecognitionSystem:
    def __init__(self, known_images_folder, model_file, config_file,
                 resize_factor=0.5, process_every_n_frames=3,
                 distance_threshold=0.55, detection_confidence=0.7):
        self.known_face_encodings, self.known_face_labels = self.load_known_faces(known_images_folder)
        self.net = self.load_cnn_model(model_file, config_file)
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.distance_threshold = distance_threshold
        self.detection_confidence = detection_confidence

        self.metrics = {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
            'total_faces': 0
        }

        self.known_unknown_faces = []
        self.unknown_faces_folder = "unknown_faces"
        os.makedirs(self.unknown_faces_folder, exist_ok=True)

    @staticmethod
    def load_known_faces(folder_path, cache_file="face_cache.pkl",
                         target_size=(300, 300), min_faces_per_image=1):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        face_encodings = []
        face_labels = []

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        app = QApplication.instance() or QApplication([])
        progress_window = ProgressBarWindow(total_images)
        progress_window.show()

        try:
            for i, filename in enumerate(image_files):
                try:
                    image_path = os.path.join(folder_path, filename)
                    image = face_recognition.load_image_file(image_path)

                    face_locs = face_recognition.face_locations(image)
                    if len(face_locs) < min_faces_per_image:
                        continue

                    encodings = face_recognition.face_encodings(image, face_locs)
                    for encoding in encodings:
                        face_encodings.append(encoding)
                        face_labels.append(os.path.splitext(filename)[0])

                    progress_window.update_progress(i + 1)
                    QApplication.processEvents()

                except Exception as e:
                    print(f"Erro processando {filename}: {str(e)}")
        finally:
            progress_window.close()
            QApplication.processEvents()

        with open(cache_file, "wb") as f:
            pickle.dump((face_encodings, face_labels), f)

        return face_encodings, face_labels

    def load_cnn_model(self, model_file, config_file):
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net

    def detect_faces(self, frame_small, rgb_small_frame, original_frame):
        h, w = frame_small.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_small, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        face_locations = []
        face_labels = []
        face_confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                scale = 1 / self.resize_factor
                original_top = int(startY * scale)
                original_right = int(endX * scale)
                original_bottom = int(endY * scale)
                original_left = int(startX * scale)

                face_locations.append((
                    original_top,
                    original_right,
                    original_bottom,
                    original_left
                ))

                label, confidence = self.recognize_face(
                    rgb_small_frame,
                    (startY, endX, endY, startX),  # Coordenadas no frame pequeno
                    original_frame,
                    (original_top, original_right, original_bottom, original_left)  # Coordenadas no frame original
                )
                face_labels.append(label)
                face_confidences.append(confidence)

        return face_locations, face_labels, face_confidences

    def recognize_face(self, rgb_small_frame, face_box, original_frame, face_location):
        face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_box])

        if not face_encodings:
            return "Desconhecido", 0.0

        matches = face_recognition.compare_faces(
            self.known_face_encodings, face_encodings[0],
            tolerance=self.distance_threshold
        )
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, face_encodings[0]
        )

        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        confidence = 1 - best_distance

        if matches[best_match_index]:
            self.metrics['true_positive'] += 1
            return self.known_face_labels[best_match_index], confidence
        else:
            self.handle_unknown_face(face_encodings[0], original_frame, face_location)
            self.metrics['false_negative'] += 1
            return "Desconhecido", confidence

    def handle_unknown_face(self, face_encoding, frame, face_location):
        if not any(np.linalg.norm(known - face_encoding) < 0.5 for known in self.known_unknown_faces):
            self.save_unknown_face(frame, face_location)
            self.known_unknown_faces.append(face_encoding)

    def save_unknown_face(self, frame, face_location):
        annotated_frame = frame.copy()
        (top, right, bottom, left) = face_location

        face_locations = [face_location]  # Lista com a localização do rosto desconhecido
        face_labels = ["Desconhecido"]  # Label do rosto desconhecido
        face_confidences = [0.0]  # Confiança (pode ser ajustada se necessário)

        annotated_frame = self.annotate_frame(annotated_frame, face_locations, face_labels, face_confidences)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.unknown_faces_folder, f"unknown_{timestamp}_{uuid.uuid4().hex}.jpg")

        cv2.imwrite(filename, annotated_frame)
        print(f"Rosto desconhecido salvo com anotações: {filename}")

        self.trigger_alert_system()

    def trigger_alert_system(self):
        try:
            self.notification = NotificationWindow()
            self.notification.show()

            QTimer.singleShot(3000, self.notification.close)

            QApplication.processEvents()
        except Exception as e:
            print(f"Erro na notificação: {str(e)}")

    def show_system_notification(self):
        try:
            notification.notify(
                title="ALERTA DE SEGURANÇA",
                message="Rosto desconhecido detectado!",
                app_name="Sistema de Reconhecimento Facial",
                timeout=10
            )
        except Exception as e:
            print(f"Erro na notificação do sistema: {str(e)}")
            print("Certifique-se de ter o plyer instalado: pip install plyer")

    def play_alert_sound(self):
        try:
            if platform.system() == "Windows":
                winsound.Beep(1000, 1000)  # Frequência 1000Hz por 1 segundo
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"])
            else:  # Linux
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"])
        except Exception as e:
            print(f"Erro no alerta sonoro: {str(e)}")
            print("\a")

    def calculate_metrics(self):
        total = self.metrics['total_faces']
        if total == 0:
            return {}

        precision = self.metrics['true_positive'] / (
                self.metrics['true_positive'] + self.metrics['false_positive'] + 1e-6
        )
        recall = self.metrics['true_positive'] / (
                self.metrics['true_positive'] + self.metrics['false_negative'] + 1e-6
        )
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': (self.metrics['true_positive'] + self.metrics['true_negative']) / total
        }

    def annotate_frame(self, frame, face_locations, face_labels, face_confidences):
        metrics = self.calculate_metrics()

        for (top, right, bottom, left), label, confidence in zip(face_locations, face_labels, face_confidences):
            color = (0, 0, 255) if label == "Desconhecido" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            text = f"{label} ({confidence:.2f})" if label != "Desconhecido" else label
            cv2.putText(frame, text, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset = 30
        for metric, value in metrics.items():
            cv2.putText(frame, f"{metric.capitalize()}: {value:.2%}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            y_offset += 30

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0

        face_locs = []
        face_labels = []
        face_confs = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_small = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            if frame_count % self.process_every_n_frames == 0:
                # Atualiza as variáveis apenas quando processar
                face_locs, face_labels, face_confs = self.detect_faces(
                    frame_small, rgb_small, frame
                )
                self.metrics['total_faces'] += len(face_locs)

            frame = self.annotate_frame(frame, face_locs, face_labels, face_confs)
            cv2.imshow("Sistema de Reconhecimento Facial Aprimorado", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = FaceRecognitionSystem(
        known_images_folder="../fotos",
        model_file="res10_300x300_ssd_iter_140000_fp16.caffemodel",
        config_file="deploy.prototxt"
    )
    system.run()