import sys
import traceback

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from rec_facial.rec_facial_cnn import FaceRecognitionSystem
from views.Cad_view import FaceRecognitionForm
from views.style import MAIN_STYLES, BUTTON_STYLES

class RecognitionThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, known_images_folder, model_file, config_file):
        super().__init__()
        self.known_images_folder = known_images_folder
        self.model_file = model_file
        self.config_file = config_file

    def run(self):
        try:
            recognition_system = FaceRecognitionSystem(
                known_images_folder=self.known_images_folder,
                model_file=self.model_file,
                config_file=self.config_file
            )
            recognition_system.run()
            self.progress.emit("Sistema de reconhecimento facial iniciado com sucesso!")
        except Exception as e:
            self.progress.emit(f"Erro: {str(e)}")

class MenuFaceRecognition(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(MAIN_STYLES)
        self.init_ui()
        self.recognition_thread = None

    def init_ui(self):
        self.setWindowTitle("Menu Principal - Reconhecimento Facial")
        self.setFixedSize(600, 400)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        title = QLabel("Sistema de Reconhecimento Facial")
        title.setStyleSheet("""
               QLabel {
                   font-size: 24px;
                   font-weight: bold;
                   color: #8ec73d;
                   padding: 10px;
                   qproperty-alignment: AlignCenter;
               }
           """)
        main_layout.addWidget(title)

        buttons = [
            ("Abrir Formulário", "primary", self.open_form),
            ("Iniciar Sistema", "primary", self.start_system),
            ("Remover Aluno", "danger", self.remove_student),
            ("Sair", "danger", self.close_application)
        ]

        for text, style_type, callback in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(BUTTON_STYLES[style_type])
            btn.clicked.connect(callback)
            btn.setMinimumHeight(40)
            main_layout.addWidget(btn)

        self.setLayout(main_layout)

        main_layout.addStretch()

    def open_form(self):
        try:
            # Fecha o menu completamente se necessário
            if hasattr(self, 'form_screen'):
                self.form_screen.close()

            # Cria nova instância mantendo a referência
            self.form_screen = FaceRecognitionForm()
            self.form_screen.setStyleSheet(MAIN_STYLES)

            # Configura relação pai-filho corretamente
            self.form_screen.setWindowModality(Qt.NonModal)
            self.form_screen.show()

            # Esconde o menu sem destruí-lo
            self.hide()

        except Exception as e:
            error_msg = f"Erro ao abrir formulário: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Erro Crítico", error_msg)

    def start_system(self):
        try:
            known_images_folder = "./fotos"
            model_file = "./rec_facial/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "./rec_facial/deploy.prototxt"

            self.recognition_thread = RecognitionThread(
                known_images_folder=known_images_folder,
                model_file=model_file,
                config_file=config_file
            )
            self.recognition_thread.progress.connect(self.show_progress)
            self.recognition_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao iniciar o sistema: {str(e)}")

    def show_progress(self, message):
        msg = QMessageBox()
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
            }
            QLabel {
                color: #333333;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #8ec73d;
                color: #ffffff;
                min-width: 80px;
                padding: 5px;
            }
        """)
        msg.information(self, "Progresso", message)

    def remove_student(self):
        msg = QMessageBox()
        msg.setStyleSheet(MAIN_STYLES)
        msg.warning(self, "Remover Aluno", "A funcionalidade de remoção será implementada.")

    def close_application(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(MAIN_STYLES)
    menu = MenuFaceRecognition()
    menu.show()
    sys.exit(app.exec_())
