import pickle
import uuid
import cv2
import numpy as np
import face_recognition
import os
import json
import time
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from views.LoadingWindow import ProgressBarWindow

# Novas importações
import platform
import subprocess
from plyer import notification
import winsound

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
            'total_faces': 0,
            'processing_times': [],
            'recognition_times': [],
            'confidences': [],
            'frame_counts': 0
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
        start_time = time.time()
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

                label, confidence, recognition_time = self.recognize_face(
                    rgb_small_frame,
                    (startY, endX, endY, startX),
                    original_frame,
                    (original_top, original_right, original_bottom, original_left)
                )
                face_labels.append(label)
                face_confidences.append(confidence)
                self.metrics['recognition_times'].append(recognition_time)

        detection_time = time.time() - start_time
        self.metrics['processing_times'].append(detection_time)
        return face_locations, face_labels, face_confidences

    def recognize_face(self, rgb_small_frame, face_box, original_frame, face_location, true_label=None):
        start_time = time.time()
        face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_box])

        if not face_encodings:
            recognition_time = time.time() - start_time
            return "Desconhecido", 0.0, recognition_time

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

        if true_label is not None:  # Modo de avaliação
            if matches[best_match_index]:
                if self.known_face_labels[best_match_index] == true_label:
                    self.metrics['true_positive'] += 1
                else:
                    self.metrics['false_positive'] += 1
            else:
                if true_label == "Desconhecido":
                    self.metrics['true_negative'] += 1
                else:
                    self.metrics['false_negative'] += 1
        else:  # Modo normal
            if matches[best_match_index]:
                self.metrics['true_positive'] += 1
            else:
                self.metrics['false_negative'] += 1

        recognition_time = time.time() - start_time

        if matches[best_match_index]:
            return self.known_face_labels[best_match_index], confidence, recognition_time
        else:
            self.handle_unknown_face(face_encodings[0], original_frame, face_location)
            return "Desconhecido", confidence, recognition_time

    def calculate_metrics(self):
        try:
            total = self.metrics['total_faces']
            processing_times = self.metrics['processing_times']
            recognition_times = self.metrics['recognition_times']
            confidences = self.metrics['confidences']

            metrics = {
                'precision': self.metrics['true_positive'] / max(1, self.metrics['true_positive'] + self.metrics[
                    'false_positive']),
                'recall': self.metrics['true_positive'] / max(1, self.metrics['true_positive'] + self.metrics[
                    'false_negative']),
                'accuracy': (self.metrics['true_positive'] + self.metrics['true_negative']) / max(1, total),
                'avg_detection_time': np.mean(processing_times) if processing_times else 0,
                'avg_recognition_time': np.mean(recognition_times) if recognition_times else 0,
                'fps': self.metrics['frame_counts'] / max(1, sum(processing_times)),
                'avg_confidence': np.mean(confidences) if confidences else 0
            }
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / max(1e-9, (
                        metrics['precision'] + metrics['recall']))
            return metrics
        except Exception as e:
            print(f"Erro ao calcular métricas: {str(e)}")
            return {}

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

    def annotate_frame(self, frame, face_locations, face_labels, face_confidences):
        # Métricas em tempo real
        metrics = self.calculate_metrics()
        y_offset = 30

        # Desenha métricas principais
        status = [
            f"FPS: {metrics.get('fps', 0):.1f}",
            f"Precisão: {metrics.get('accuracy', 0):.2%}",
            f"Confiança Média: {metrics.get('avg_confidence', 0):.2f}",
            f"Tempo/Frame: {metrics.get('avg_detection_time', 0):.3f}s"
        ]

        for text in status:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30

        # Desenha retângulos e labels
        for (top, right, bottom, left), label, confidence in zip(face_locations, face_labels, face_confidences):
            color = (0, 0, 255) if label == "Desconhecido" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            text = f"{label} ({confidence:.2f})" if label != "Desconhecido" else label
            cv2.putText(frame, text, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def generate_report(self, filename="performance_report.json"):
        """Gera um relatório completo das métricas em JSON e no console"""
        try:
            report = self.calculate_metrics()

            # Adiciona métricas adicionais ao relatório
            report.update({
                'total_faces': self.metrics['total_faces'],
                'processing_frames': self.metrics['frame_counts'],
                'unknown_faces_detected': len(self.known_unknown_faces)
            })

            # Salva em arquivo
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)

            return report

        except Exception as e:
            print(f"Erro ao gerar relatório: {str(e)}")
            return {}

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()

        face_locs = []
        face_labels = []
        face_confs = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_time = time.time()

                # Processamento do frame
                frame_small = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

                if frame_count % self.process_every_n_frames == 0:
                    face_locs, face_labels, face_confs = self.detect_faces(frame_small, rgb_small, frame)
                    if face_locs:  # Só atualiza se faces forem detectadas
                        self.metrics['total_faces'] += len(face_locs)
                        self.metrics['confidences'].extend(face_confs)

                # Atualiza métricas de tempo
                self.metrics['frame_counts'] += 1
                self.metrics['processing_times'].append(time.time() - frame_time)

                # Exibe frame com métricas
                frame = self.annotate_frame(frame, face_locs, face_labels, face_confs)
                cv2.imshow("Sistema de Reconhecimento Facial - Métricas em Tempo Real", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Garante a execução destes comandos mesmo com erro
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Limpeza adicional do OpenCV

            report = self.generate_report()
            if report:
                print("\n--- Relatório Final da Sessão ---")
                print(f"Tempo Total: {time.time() - start_time:.2f}s")
                print(f"Frames Processados: {self.metrics['frame_counts']}")
                print(f"Acurácia: {report.get('accuracy', 0):.2%}")
                print(f"Precisão: {report.get('precision', 0):.2%}")
                print(f"Recall: {report.get('recall', 0):.2%}")
                print(f"F1-Score: {report.get('f1_score', 0):.2%}")
                print(f"FPS Médio: {report.get('fps', 0):.1f}")
                print(f"Confiança Média: {report.get('avg_confidence', 0):.2f}")
            else:
                print("Não foi possível gerar o relatório!")


if __name__ == "__main__":
    system = FaceRecognitionSystem(
        known_images_folder="../fotos",
        model_file="res10_300x300_ssd_iter_140000_fp16.caffemodel",
        config_file="deploy.prototxt"
    )
    system.run()