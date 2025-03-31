import cv2
import sqlite3
import os
from pathlib import Path
from views.style import MAIN_STYLES, BUTTON_STYLES, PREVIEW_STYLES

from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                             QHBoxLayout, QMessageBox, QFileDialog)


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.current_frame = None

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            while self.running:
                ret, frame = cap.read()
                if ret:
                    self.current_frame = frame
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_ready.emit(qt_image)
            cap.release()
        except Exception as e:
            self.error_occurred.emit(f"Erro na câmera: {str(e)}")

    def stop(self):
        self.running = False


class FaceRecognitionForm(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(MAIN_STYLES)
        self.camera_thread = None
        self.current_cv_frame = None
        self.photo_path = ""
        self.setup_ui()
        self.setup_database()

    def setup_ui(self):
        self.setWindowTitle("Captura de Foto com Reconhecimento Facial")
        self.setFixedSize(800, 700)

        main_container = QVBoxLayout()
        main_container.setContentsMargins(20, 20, 20, 20)
        main_container.setSpacing(15)

        form_container = QVBoxLayout()
        form_container.setSpacing(10)

        self.name_label = QLabel("Nome completo:")
        self.name_input = QLineEdit()
        self.id_label = QLabel("Matrícula:")
        self.id_input = QLineEdit()

        form_container.addWidget(self.name_label)
        form_container.addWidget(self.name_input)
        form_container.addWidget(self.id_label)
        form_container.addWidget(self.id_input)

        preview_container = QVBoxLayout()
        preview_container.setSpacing(10)

        preview_title = QLabel("Área de Captura")
        preview_title.setStyleSheet("font-size: 14pt; color: #4a4a4a;")
        preview_title.setAlignment(Qt.AlignCenter)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("preview_label")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)

        preview_container.addWidget(preview_title)
        preview_container.addWidget(self.preview_label)

        camera_controls = QHBoxLayout()
        camera_controls.setSpacing(15)
        camera_controls.addStretch()

        self.btn_start = QPushButton("Iniciar Câmera")
        self.btn_stop = QPushButton("Parar Câmera")
        self.btn_capture = QPushButton("Capturar Foto")
        self.btn_load = QPushButton("Carregar Foto")

        camera_controls.addWidget(self.btn_start)
        camera_controls.addWidget(self.btn_stop)
        camera_controls.addWidget(self.btn_capture)
        camera_controls.addWidget(self.btn_load)
        camera_controls.addStretch()

        global_actions = QHBoxLayout()
        global_actions.addStretch()

        self.btn_save = QPushButton("Salvar Cadastro")
        self.btn_back = QPushButton("Voltar ao Menu")

        global_actions.addWidget(self.btn_save)
        global_actions.addWidget(self.btn_back)

        main_container.addLayout(form_container, 20)  # 20% do espaço
        main_container.addLayout(preview_container, 50)  # 50% do espaço
        main_container.addLayout(camera_controls, 15)  # 15% do espaço
        main_container.addLayout(global_actions, 15)  # 15% do espaço

        self.preview_label.setObjectName("preview_label")
        self.preview_label.setStyleSheet(PREVIEW_STYLES["inactive"])
        self.apply_button_styles()

        self.setLayout(main_container)
        self.setup_connections()
        self.update_buttons_state(False)

    def apply_button_styles(self):
        self.btn_start.setStyleSheet(BUTTON_STYLES["primary"])
        self.btn_stop.setStyleSheet(BUTTON_STYLES["danger"])
        self.btn_capture.setStyleSheet(BUTTON_STYLES["secondary"])
        self.btn_save.setStyleSheet(BUTTON_STYLES["primary"])
        self.btn_load.setStyleSheet(BUTTON_STYLES["secondary"])
        self.btn_back.setStyleSheet(BUTTON_STYLES["danger"])

    def setup_connections(self):
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_photo)
        self.btn_load.clicked.connect(self.load_photo)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_back.clicked.connect(self.close)

    def setup_database(self):
        Path("database").mkdir(exist_ok=True)
        self.conn = sqlite3.connect("database/faces.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cadastros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                matricula TEXT UNIQUE NOT NULL,
                foto_path TEXT NOT NULL
            )
        """)

    def update_buttons_state(self, camera_active):
        self.btn_start.setEnabled(not camera_active)
        self.btn_stop.setEnabled(camera_active)
        self.btn_capture.setEnabled(camera_active)

    def start_camera(self):
        self.preview_label.setStyleSheet(PREVIEW_STYLES["active"])

        if not self.validate_fields():
            return

        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_preview)
        self.camera_thread.error_occurred.connect(self.show_error)
        self.camera_thread.start()
        self.update_buttons_state(True)

    def stop_camera(self):
        self.preview_label.setStyleSheet(PREVIEW_STYLES["inactive"])
        if self.camera_thread:
            try:
                self.camera_thread.frame_ready.disconnect()
                self.camera_thread.error_occurred.disconnect()
            except TypeError:
                pass

            self.camera_thread.stop()
            self.camera_thread.quit()
            self.camera_thread.wait(1000)  # Timeout de 1 segundo

            if hasattr(self, 'last_captured_frame') and self.last_captured_frame:
                self.preview_label.setPixmap(self.last_captured_frame)
            else:
                self.preview_label.setText("Câmera desativada\nPronto para novas ações")
                self.preview_label.setStyleSheet("""
                       background-color: #f0f0f0; 
                       color: #666;
                       font: 12pt 'Arial';
                       qproperty-alignment: AlignCenter;
                   """)

            self.camera_thread = None
            self.update_buttons_state(False)

    def update_preview(self, image):
        pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def capture_photo(self):
        if self.camera_thread and self.camera_thread.current_frame is not None:
            matricula = self.id_input.text().strip()

            if not matricula:
                self.show_error("Digite a matrícula antes de capturar a foto!")
                return

            try:
                Path("fotos").mkdir(exist_ok=True)
                self.photo_path = f"fotos/{matricula}.jpg"

                cv2.imwrite(self.photo_path, self.camera_thread.current_frame)

                pixmap = QPixmap(self.photo_path)
                self.preview_label.setPixmap(
                    pixmap.scaled(
                        self.preview_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )

                self.last_captured_frame = pixmap

                QMessageBox.information(
                    self,
                    "Foto Capturada",
                    f"Foto salva em:\n{self.photo_path}"
                )

            except Exception as e:
                self.show_error(f"Erro ao salvar foto: {str(e)}")

        else:
            self.show_error("Câmera não iniciada ou não há frame disponível!")

    def load_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Foto", "", "Imagens (*.jpg *.png)"
        )
        if path:
            self.photo_path = path
            self.preview_label.setPixmap(QPixmap(path).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def validate_fields(self):
        if not self.name_input.text().strip():
            self.show_error("O nome é obrigatório!")
            return False
        if not self.id_input.text().strip():
            self.show_error("A matrícula é obrigatória!")
            return False
        return True

    def save_data(self):
        if not self.validate_fields() or not self.photo_path:
            self.show_error("Complete todos os campos e capture/carregue uma foto!")
            return

        try:
            self.conn.execute(
                "INSERT INTO cadastros (nome, matricula, foto_path) VALUES (?, ?, ?)",
                (self.name_input.text(), self.id_input.text(), self.photo_path)
            )
            self.conn.commit()
            QMessageBox.information(self, "Sucesso", "Cadastro salvo com sucesso!")
            self.reset_form()
        except sqlite3.IntegrityError:
            self.show_error("Matrícula já cadastrada!")
        except Exception as e:
            self.show_error(f"Erro ao salvar: {str(e)}")

    def reset_form(self):
        self.name_input.clear()
        self.id_input.clear()
        self.preview_label.clear()
        self.photo_path = ""
        self.stop_camera()

    def show_error(self, message):
        QMessageBox.critical(self, "Erro", message)

    def closeEvent(self, event):
        self.stop_camera()
        self.conn.close()
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = FaceRecognitionForm()
    window.show()
    sys.exit(app.exec_())