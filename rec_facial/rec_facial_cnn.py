import pickle
import uuid
import cv2
import numpy as np
import face_recognition
import os
from PyQt5.QtWidgets import QApplication
from views.LoadingWindow import ProgressBarWindow

class FaceRecognitionSystem:
    def __init__(self, known_images_folder, model_file, config_file, resize_factor=0.3, process_every_n_frames=5,
                 distance_threshold=0.6):
        self.known_face_encodings, self.known_face_labels = self.load_known_faces(known_images_folder)
        self.net = self.load_cnn_model(model_file, config_file)
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames
        self.distance_threshold = distance_threshold

        self.frame_count = 0
        self.face_locations = []
        self.face_labels = []

        self.true_positive_count = 0
        self.false_negative_count = 0
        self.false_positive_count = 0
        self.true_negative_count = 0

        self.known_unknown_faces = []

    @staticmethod
    def load_known_faces(folder_path, cache_file="face_cache.pkl", resize_width=300, resize_height=300):
        # Verifica se o cache já existe para carregar codificações
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                print("Carregando codificações do cache...")
                return pickle.load(f)

        print("Codificações não encontradas no cache. Processando imagens...")
        face_encodings = []
        face_labels = []

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        # Verifica se já existe uma instância do QApplication
        app_instance = QApplication.instance()
        if not app_instance:
            app = QApplication([])  # Cria uma nova instância do QApplication se necessário
        else:
            app = app_instance

        # Configura a barra de progresso
        progress_window = ProgressBarWindow(total_images)
        progress_window.show()

        try:
            for i, file_name in enumerate(image_files):
                try:
                    image_path = os.path.join(folder_path, file_name)
                    image = face_recognition.load_image_file(image_path)

                    # Redimensiona a imagem para processar mais rapidamente
                    small_image = cv2.resize(image, (resize_width, resize_height))
                    encodings = face_recognition.face_encodings(small_image)

                    # Armazena a codificação e o rótulo se houver uma face detectada
                    if encodings:
                        face_encodings.append(encodings[0])
                        face_labels.append(os.path.splitext(file_name)[0])

                    # Atualiza a barra de progresso
                    progress_window.update_progress(i + 1)
                    QApplication.processEvents()  # Atualiza a interface gráfica durante o loop

                except Exception as e:
                    print(f"Erro ao processar a imagem {file_name}: {e}")
        finally:
            # Em vez de fechar diretamente aqui, emitimos um sinal ou retornamos a indicação de término
            progress_window.hide()  # Apenas esconde a barra para evitar interações
            QApplication.processEvents()  # Garante atualização final da interface gráfica

        # Salva as codificações no cache
        with open(cache_file, "wb") as f:
            pickle.dump((face_encodings, face_labels), f)
            print("Codificações salvas no cache.")

        return face_encodings, face_labels

    @staticmethod
    def load_cnn_model(model_file, config_file):
        return cv2.dnn.readNetFromCaffe(config_file, model_file)

    def process_frame(self, frame):
        frame_small = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        rgb_small_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        return frame_small, rgb_small_frame

    def detect_faces(self, frame_small, rgb_small_frame, frame):
        h, w = frame_small.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_small, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        self.face_locations.clear()
        self.face_labels.clear()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                scaled_box = (
                    int(startY / self.resize_factor),
                    int(endX / self.resize_factor),
                    int(endY / self.resize_factor),
                    int(startX / self.resize_factor)
                )
                self.face_locations.append(scaled_box)
                self.recognize_face(rgb_small_frame, (startY, endX, endY, startX), frame)

    def recognize_face(self, rgb_small_frame, face_box, frame):
        face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_box])

        if face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < self.distance_threshold:
                    self.face_labels.append(self.known_face_labels[best_match_index])
                    self.true_positive_count += 1
                else:
                    self.handle_unknown_face(frame, face_box, face_encodings[0])
            else:
                self.handle_unknown_face(frame, face_box, face_encodings[0])
        else:
            self.face_labels.append("Desconhecido")
            self.true_negative_count += 1

    def handle_unknown_face(self, frame, face_box, face_encoding):
        top, right, bottom, left = face_box

        self.face_labels.append("Desconhecido")
        self.false_negative_count += 1

        if not self.is_known_unknown_face(face_encoding):
            self.save_unknown_face(face_encoding, frame)

    def is_known_unknown_face(self, face_encoding, threshold=0.6):
        for known_encoding in self.known_unknown_faces:
            distance = np.linalg.norm(known_encoding - face_encoding)
            if distance < threshold:
                return True
        return False

    def save_unknown_face(self, face_encoding, frame):
        self.known_unknown_faces.append(face_encoding)

        unknown_faces_folder = "unknown_faces"
        os.makedirs(unknown_faces_folder, exist_ok=True)

        self.annotate_frame(frame)

        file_name = os.path.join(unknown_faces_folder, f"unknown_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(file_name, frame)
        print(f"Frame anotado com rosto desconhecido salvo em: {file_name}")

    def annotate_frame(self, frame):
        for (top, right, bottom, left), label in zip(self.face_locations, self.face_labels):
            color = (0, 0, 255) if label == "Desconhecido" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        accuracy = self.calculate_accuracy()
        cv2.putText(
            frame, f"Acur\u00e1cia: {accuracy:.2%}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    def calculate_accuracy(self):
        total_predictions = (
                self.true_positive_count +
                self.false_negative_count +
                self.false_positive_count +
                self.true_negative_count
        )
        if total_predictions > 0:
            return (self.true_positive_count + self.true_negative_count) / total_predictions
        return 0

    def run(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            self.frame_count += 1

            frame_small, rgb_small_frame = self.process_frame(frame)

            if self.frame_count % self.process_every_n_frames == 0:
                self.detect_faces(frame_small, rgb_small_frame, frame)

            self.annotate_frame(frame)

            cv2.imshow("Reconhecimento Facial com CNN", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem(
        known_images_folder="../fotos",
        model_file="res10_300x300_ssd_iter_140000_fp16.caffemodel",
        config_file="deploy.prototxt"
    )
    face_recognition_system.run()
