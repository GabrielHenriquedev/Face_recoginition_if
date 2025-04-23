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
import platform
import subprocess
from plyer import notification
import winsound
from views.notification import NotificationWindow


class FaceRecognitionSystem:
    def __init__(self, known_images_folder, model_file, config_file,
                 resize_factor=0.5, distance_threshold=0.55, detection_confidence=0.7):
        self.known_face_encodings, self.known_face_labels = self.load_known_faces(known_images_folder)
        self.net = self.load_cnn_model(model_file, config_file)
        self.resize_factor = resize_factor
        self.distance_threshold = distance_threshold
        self.detection_confidence = detection_confidence

        # Parâmetros de temporização
        self.detection_interval = 0.75
        self.recognition_interval = 1.5
        self.last_detection_time = 0
        self.last_recognition_time = 0
        self.max_detection_age = 1.5

        # Suavização visual
        self.smoothed_locations = []
        self.smoothing_factor = 0.5

        self.metrics = {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
            'total_faces': 0,
            'detection_times': [],
            'recognition_times': [],
            'recognition_confidences': [],
            'detection_confidences': [],
            'start_time': time.time(),
            'total_frames': 0,
            'frame_counts': 0
        }

        self.known_unknown_faces = []
        self.unknown_faces_folder = "unknown_faces"
        os.makedirs(self.unknown_faces_folder, exist_ok=True)
        self.current_fps = 0.0

    @staticmethod
    def load_known_faces(folder_path, cache_file="face_cache.pkl", min_faces_per_image=1):
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
        # Tenta configurar para CUDA, fallback para CPU se falhar
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def detect_faces(self, frame_small):
        h, w = frame_small.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_small, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        face_locations = []
        detection_confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                scale = 1 / self.resize_factor
                face_locations.append((
                    int(startY * scale),
                    int(endX * scale),
                    int(endY * scale),
                    int(startX * scale)
                ))
                detection_confidences.append(confidence)

        return face_locations, detection_confidences

    def recognize_faces(self, rgb_small_frame, face_locations, detection_confidences):
        face_labels = []
        recognition_confidences = []

        for idx, face_box in enumerate(face_locations):
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_box])
                if not face_encodings:
                    recognition_confidences.append(0.0)
                    face_labels.append("Desconhecido")
                    continue

                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encodings[0],
                    tolerance=self.distance_threshold
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encodings[0]
                )
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                confidence = 1 - best_distance

                if matches[best_match_index]:
                    label = self.known_face_labels[best_match_index]
                    self.metrics['true_positive'] += 1
                else:
                    label = "Desconhecido"
                    # Usa o encoding facial para verificação
                    if self.face_should_be_known(face_encodings[0]):
                        self.metrics['false_negative'] += 1
                    else:
                        self.metrics['true_negative'] += 1

                    self.handle_unknown_face(
                        face_encodings[0],
                        face_box,
                        detection_confidences[idx],
                        confidence
                    )

                face_labels.append(label)
                recognition_confidences.append(confidence)
                self.metrics['recognition_confidences'].append(confidence)
                self.metrics['total_faces'] += 1

            except Exception as e:
                print(f"Erro no reconhecimento: {str(e)}")
                face_labels.append("Erro")
                recognition_confidences.append(0.0)

        return face_labels, recognition_confidences

    def execute_recognition_cycle(self):
        """Executa um ciclo completo de detecção e reconhecimento"""
        try:
            # Passo 1: Detecção se necessário
            if (time.time() - self.last_detection_time) > self.detection_interval:
                frame_small = cv2.resize(self.current_frame, (0, 0),
                                         fx=self.resize_factor,
                                         fy=self.resize_factor)
                new_locs, new_confs = self.detect_faces(frame_small)

                if new_locs:
                    self.current_face_locations = new_locs
                    self.current_detection_confidences = new_confs
                    self.last_detection_time = time.time()


            # Passo 2: Reconhecimento sempre com o frame mais recente
            current_time = time.time()
            if (current_time - self.last_recognition_time) >= self.recognition_interval:
                if self.current_face_locations:
                    frame_small = cv2.resize(self.current_frame, (0, 0),
                                             fx=self.resize_factor,
                                             fy=self.resize_factor)
                    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

                    # Converter coordenadas
                    small_locs = [
                        (
                            int(top * self.resize_factor),
                            int(right * self.resize_factor),
                            int(bottom * self.resize_factor),
                            int(left * self.resize_factor)
                        ) for (top, right, bottom, left) in self.current_face_locations
                    ]

                    self.current_face_labels, self.current_recognition_confidences = self.recognize_faces(
                        rgb_small, small_locs, self.current_detection_confidences
                    )

                self.last_recognition_time = current_time

        except Exception as e:
            print(f"Erro no ciclo de reconhecimento: {str(e)}")

    def face_should_be_known(self, face_encoding):
        """Verifica se o rosto deveria ser reconhecido com base nos encodings conhecidos"""
        verification_threshold = self.distance_threshold * 0.7  # Threshold mais rigoroso
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        return any(d <= verification_threshold for d in distances)

    def update_smoothed_locations(self, new_locations):
        """Aplica suavização exponencial nas coordenadas dos rostos"""
        if not self.smoothed_locations:
            self.smoothed_locations = new_locations.copy()
            return

        # Match mais próximo entre detecções atuais e suavizadas
        matched = []
        for new_loc in new_locations:
            distances = [np.linalg.norm(np.array(new_loc) - np.array(smoothed))
                         for smoothed in self.smoothed_locations]
            if distances:
                idx = np.argmin(distances)
                matched.append((idx, new_loc))

        # Atualizar posições suavizadas
        updated = []
        for idx, new_loc in matched:
            smoothed = [
                int(self.smoothed_locations[idx][i] * (1 - self.smoothing_factor) +
                    new_loc[i] * self.smoothing_factor)
                for i in range(4)
            ]
            updated.append(smoothed)

        self.smoothed_locations = updated

    def handle_unknown_face(self, face_encoding, face_location, det_conf, rec_conf):
        if not any(np.linalg.norm(known - face_encoding) < 0.5 for known in self.known_unknown_faces):
            self.save_unknown_face(face_location, det_conf, rec_conf)
            self.known_unknown_faces.append(face_encoding)

    def save_unknown_face(self, face_location, det_conf, rec_conf):
        # 1. Obter frame original atual
        original_frame = self.current_frame.copy()

        # 2. Converter coordenadas para o tamanho original
        (top, right, bottom, left) = (
            int(face_location[0] / self.resize_factor),
            int(face_location[1] / self.resize_factor),
            int(face_location[2] / self.resize_factor),
            int(face_location[3] / self.resize_factor)
        )

        # 3. Garantir limites válidos
        height, width = original_frame.shape[:2]
        top = max(0, min(top, height))
        bottom = max(0, min(bottom, height))
        left = max(0, min(left, width))
        right = max(0, min(right, width))

        # 4. Criar frame anotado completo
        annotated_frame = self.annotate_frame(
            frame=original_frame,
            face_locations=[(top, right, bottom, left)],
            face_labels=["Desconhecido"],
            detection_confidences=[det_conf],
            recognition_confidences=[rec_conf],
            fps=self.current_fps
        )

        # 5. Salvar frame completo com anotações
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.unknown_faces_folder,
                                f"unknown_full_{timestamp}_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(filename, annotated_frame)

        # 6. (Opcional) Salvar também close-up do rosto
        face_roi = original_frame[top:bottom, left:right]
        if face_roi.size > 0:
            cv2.imwrite(filename.replace("_full_", "_closeup_"), face_roi)

        print(f"Frame completo salvo: {filename}")

    def trigger_alert_system(self):
        try:
            self.notification = NotificationWindow()
            self.notification.show()
            QTimer.singleShot(3000, self.notification.close)
            QApplication.processEvents()
            self.show_system_notification()
            self.play_alert_sound()
        except Exception as e:
            print(f"Erro na notificação: {str(e)}")

    def annotate_frame(self, frame, face_locations, face_labels,
                       detection_confidences, recognition_confidences, fps):
        # Calcular métricas atualizadas
        metrics = self.calculate_metrics()

        # Criar texto das métricas
        metrics_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Conf Média: {metrics['avg_confidence']:.2f}",
            f"Acurácia: {metrics['accuracy']:.2%}",
            f"Faces: {metrics['faces_detected']}"
        ]

        # Adicionar texto no frame
        y_offset = 30
        for text in metrics_text:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        # Desenhar retângulos e labels dos rostos
        for (top, right, bottom, left), label, det_conf, rec_conf in zip(
                face_locations, face_labels, detection_confidences, recognition_confidences):
            # Escolher cor baseado no reconhecimento
            color = (0, 0, 255) if label == "Desconhecido" else (0, 255, 0)

            # Desenhar retângulo
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Criar texto do label
            label_text = f"{label} ({rec_conf:.2f})"

            # Calcular tamanho do texto para fundo
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Desenhar fundo do texto
            cv2.rectangle(frame,
                          (left, bottom - text_height - 4),
                          (left + text_width, bottom),
                          color, cv2.FILLED)

            # Escrever texto
            cv2.putText(frame, label_text,
                        (left + 4, bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def calculate_metrics(self):
        # Implementação completa das métricas
        total_time = time.time() - self.metrics['start_time']

        tp = self.metrics['true_positive']
        fp = self.metrics['false_positive']
        tn = self.metrics['true_negative']
        fn = self.metrics['false_negative']

        # Prevenir divisão por zero
        total = (self.metrics['true_positive'] +
                 self.metrics['false_positive'] +
                 self.metrics['true_negative'] +
                 self.metrics['false_negative'])

        # Cálculos com tratamento de divisão por zero
        accuracy = (self.metrics['true_positive'] + self.metrics['true_negative']) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'total_time': round(total_time, 2),
            'frames_processed': self.metrics['total_frames'],
            'accuracy': accuracy,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'avg_confidence': round(
                np.mean(self.metrics['recognition_confidences']) if self.metrics['recognition_confidences'] else 0, 4),
            'faces_detected': self.metrics['total_faces'],
            'fps': round(self.metrics['total_frames'] / total_time, 2) if total_time > 0 else 0
        }

    def run(self, video_path="meu_video.mp4"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir o vídeo.")
            return

        # Inicializa métricas
        self.metrics['start_time'] = time.time()
        self.metrics['total_frames'] = 0
        frame_count = 0
        start_time = time.time()

        # Estado do sistema
        self.current_face_locations = []
        self.current_detection_confidences = []
        self.current_face_labels = []
        self.current_recognition_confidences = []

        try:
            while True:
                ret, self.current_frame = cap.read()
                if not ret:
                    break

                self.metrics['total_frames'] += 1
                frame_count += 1

                # Executa o ciclo de reconhecimento
                self.execute_recognition_cycle()

                if (time.time() - start_time) >= 1.0:
                    self.current_fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

                # Anotação do frame
                annotated_frame = self.annotate_frame(
                    frame=self.current_frame.copy(),
                    face_locations=self.smoothed_locations if self.smoothed_locations else self.current_face_locations,
                    face_labels=self.current_face_labels,
                    detection_confidences=self.current_detection_confidences,
                    recognition_confidences=self.current_recognition_confidences,
                    fps=self.current_fps
                )

                cv2.imshow("Sistema de Reconhecimento Facial", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Encerrando pela tecla 'q'...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.generate_report()

    def generate_report(self, filename="performance_report.json"):
        metrics = self.calculate_metrics()

        report = {
            'performance_metrics': metrics,
            'hardware_info': {
                'os': platform.system(),
                'processor': platform.processor(),
                'opencv_version': cv2.__version__,
            }
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)

        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)


if __name__ == "__main__":
    system = FaceRecognitionSystem(
        known_images_folder="../fotos",
        model_file="res10_300x300_ssd_iter_140000_fp16.caffemodel",
        config_file="deploy.prototxt"
    )
    system.run()