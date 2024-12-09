import cv2
import numpy as np
import face_recognition
import os


class FaceRecognitionSystem:
    def __init__(self, known_images_folder, model_file, config_file, resize_factor=0.3, process_every_n_frames=5):
        # Inicialização dos parâmetros
        self.known_face_encodings, self.known_face_labels = self.load_known_faces(known_images_folder)
        self.net = self.load_cnn_model(model_file, config_file)
        self.resize_factor = resize_factor
        self.process_every_n_frames = process_every_n_frames

        # Variáveis de controle
        self.frame_count = 0
        self.face_locations = []
        self.face_labels = []

        # Contadores para precisão
        self.true_positive_count = 0
        self.false_negative_count = 0
        self.false_positive_count = 0
        self.true_negative_count = 0

    @staticmethod
    def load_known_faces(folder_path):
        """Carrega todas as imagens da pasta e calcula as codificações faciais."""
        face_encodings = []
        face_labels = []

        for file_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    face_encodings.append(encodings[0])
                    face_labels.append(os.path.splitext(file_name)[0])  # Usa o nome do arquivo como rótulo

        return face_encodings, face_labels

    @staticmethod
    def load_cnn_model(model_file, config_file):
        """Carrega o modelo pré-treinado da CNN do OpenCV."""
        return cv2.dnn.readNetFromCaffe(config_file, model_file)

    def process_frame(self, frame):
        """Redimensiona o frame e converte para RGB."""
        frame_small = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        rgb_small_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        return frame_small, rgb_small_frame

    def detect_faces(self, frame_small, rgb_small_frame):
        """Detecta faces no frame usando a CNN e verifica o reconhecimento."""
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

                # Ajustar as coordenadas para o frame original
                scaled_box = (
                    int(startY / self.resize_factor),  # top
                    int(endX / self.resize_factor),    # right
                    int(endY / self.resize_factor),    # bottom
                    int(startX / self.resize_factor)   # left
                )
                self.face_locations.append(scaled_box)

                # Reconhecimento facial
                self.recognize_face(rgb_small_frame, (startY, endX, endY, startX))

    def recognize_face(self, rgb_small_frame, face_box):
        """Reconhece uma face detectada usando as codificações faciais conhecidas."""
        face_encodings = face_recognition.face_encodings(rgb_small_frame, [face_box])

        # Verifica se há codificação de face
        if face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encodings[0])

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    self.face_labels.append(self.known_face_labels[best_match_index])
                    self.true_positive_count += 1
                else:
                    self.face_labels.append("Desconhecido")
                    self.false_negative_count += 1
            else:
                self.face_labels.append("Desconhecido")
                self.false_negative_count += 1
        else:
            self.face_labels.append("Desconhecido")
            self.true_negative_count += 1

    def annotate_frame(self, frame):
        """Desenha os retângulos e rótulos no frame."""
        for (top, right, bottom, left), label in zip(self.face_locations, self.face_labels):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Verde para faces detectadas
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Exibe a acurácia no frame
        accuracy = self.calculate_accuracy()
        cv2.putText(
            frame, f"Acurácia: {accuracy:.2%}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    def calculate_accuracy(self):
        """Calcula a acurácia baseada nos contadores."""
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
        """Executa o sistema de reconhecimento facial."""
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            self.frame_count += 1

            frame_small, rgb_small_frame = self.process_frame(frame)

            if self.frame_count % self.process_every_n_frames == 0:
                self.detect_faces(frame_small, rgb_small_frame)

            self.annotate_frame(frame)

            cv2.imshow("Reconhecimento Facial com CNN", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


# Inicializar e executar o sistema
if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem(
        known_images_folder="../fotos",
        model_file="res10_300x300_ssd_iter_140000_fp16.caffemodel",
        config_file="deploy.prototxt"
    )
    face_recognition_system.run()
