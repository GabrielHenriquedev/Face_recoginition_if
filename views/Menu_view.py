import sys
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox, QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from rec_facial.rec_facial_cnn import FaceRecognitionSystem
from views.Cad_view import FaceRecognitionForm


class RecognitionThread(QThread):
    progress = pyqtSignal(str)  # Para emitir mensagens de progresso

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
        self.setWindowTitle("Menu Principal")
        self.setGeometry(100, 100, 400, 300)

        # Configuração do layout
        layout = QVBoxLayout()

        # Rótulo do menu
        label = QLabel("Bem-vindo ao Menu Principal")
        layout.addWidget(label)

        # Botão para abrir o formulário
        open_form_button = QPushButton("Abrir Formulário")
        open_form_button.clicked.connect(self.open_form)
        layout.addWidget(open_form_button)

        # Botão para iniciar o sistema
        start_system_button = QPushButton("Iniciar Sistema")
        start_system_button.clicked.connect(self.start_system)
        layout.addWidget(start_system_button)

        # Botão para remover um aluno
        remove_student_button = QPushButton("Remover Aluno")
        remove_student_button.clicked.connect(self.remove_student)
        layout.addWidget(remove_student_button)

        # Botão para sair
        exit_button = QPushButton("Sair")
        exit_button.clicked.connect(self.close_application)
        layout.addWidget(exit_button)

        self.setLayout(layout)

        # Cria a thread de reconhecimento facial
        self.recognition_thread = None

    def open_form(self):
        try:
            self.form_screen = FaceRecognitionForm(parent_menu=self)
            self.form_screen.show()
            self.hide()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível abrir o formulário: {str(e)}")

    def start_system(self):
        try:
            known_images_folder = "./fotos"
            model_file = "./rec_facial/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "./rec_facial/deploy.prototxt"

            # Inicia a thread para o reconhecimento facial
            self.recognition_thread = RecognitionThread(
                known_images_folder=known_images_folder,
                model_file=model_file,
                config_file=config_file
            )
            self.recognition_thread.progress.connect(self.show_progress)
            self.recognition_thread.start()  # Inicia a execução na thread separada
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao iniciar o sistema: {str(e)}")

    def show_progress(self, message):
        """Exibe as mensagens de progresso na interface gráfica."""
        QMessageBox.information(self, "Progresso", message)

    def remove_student(self):
        QMessageBox.warning(self, "Remover Aluno", "A funcionalidade de remoção será implementada.")

    def close_application(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MenuFaceRecognition()
    menu.show()
    sys.exit(app.exec_())
