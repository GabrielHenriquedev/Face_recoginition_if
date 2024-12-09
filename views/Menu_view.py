from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox

from rec_facial.rec_facial_cnn import FaceRecognitionSystem
from views.Cad_view import FaceRecognitionForm


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
        open_form_button.clicked.connect(self.open_form)  # Conectado no AppManager
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

    def open_form(self):
        """
        Abre a tela de formulário e esconde o menu principal.
        """
        try:
            self.form_screen = FaceRecognitionForm(parent_menu=self)  # Cria a tela do formulário com referência ao menu
            self.form_screen.show()  # Mostra a tela do formulário
            self.hide()  # Esconde o menu principal
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível abrir o formulário: {str(e)}")

    def start_system(self):
        """
        Inicia o sistema de reconhecimento facial.
        """
        try:
            known_images_folder = "./fotos"
            model_file = "./rec_facial/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "./rec_facial/deploy.prototxt"

            recognition_system = FaceRecognitionSystem(
                known_images_folder=known_images_folder,
                model_file=model_file,
                config_file=config_file
            )
            recognition_system.run()

            QMessageBox.information(self, "Sistema Iniciado",
                                    "O sistema de reconhecimento facial foi iniciado com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao iniciar o sistema: {str(e)}")

    def remove_student(self):
        """
        Remove um aluno (com lógica posterior).
        """
        QMessageBox.warning(self, "Remover Aluno", "A funcionalidade de remoção será implementada.")

    def close_application(self):
        """
        Fecha a aplicação.
        """
        self.close()
