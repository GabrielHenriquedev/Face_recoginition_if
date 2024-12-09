import sqlite3

from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from controllers.Cad_controller import browse_photo
import cv2

class FaceRecognitionForm(QWidget):
    def __init__(self, parent_menu=None):
        super().__init__()

        self.parent_menu = parent_menu  # Armazena a referência ao menu principal
        self.setWindowTitle("Formulário de Reconhecimento Facial")
        self.setGeometry(100, 100, 400, 200)

        # Criando os widgets do formulário
        self.name_label = QLabel("Nome:")
        self.name_input = QLineEdit()

        self.id_label = QLabel("Matrícula:")
        self.id_input = QLineEdit()

        self.btn_foto = QPushButton("Capturar Foto")
        self.btn_salvar = QPushButton("Salvar")

        self.browse_button = QPushButton("Selecionar Foto")
        self.browse_button.clicked.connect(lambda: browse_photo(self))

        # Layout para cada linha do formulário
        form_layout = QVBoxLayout()

        # Linha Nome
        name_layout = QHBoxLayout()
        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_input)
        form_layout.addLayout(name_layout)

        # Linha Matrícula
        id_layout = QHBoxLayout()
        id_layout.addWidget(self.id_label)
        id_layout.addWidget(self.id_input)
        form_layout.addLayout(id_layout)

        # Botões de captura e salvar
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_foto)
        button_layout.addWidget(self.btn_salvar)
        form_layout.addLayout(button_layout)

        self.btn_foto.clicked.connect(self.capturar_foto)
        self.btn_salvar.clicked.connect(self.salvar_dados)

        # Botão para voltar ao menu
        back_button = QPushButton("Voltar ao Menu")
        back_button.clicked.connect(self.return_to_menu)
        form_layout.addWidget(back_button)

        # Definindo o layout principal
        self.setLayout(form_layout)

    def return_to_menu(self):
        """
        Retorna ao menu principal.
        """
        self.hide()  # Esconde o formulário
        if self.parent_menu:
            self.parent_menu.show()  # Mostra o menu principal

    def capturar_foto(self):
        """
        Captura a foto usando a câmera e salva no diretório fotos/.
        """
        # Verifica se a matrícula foi informada
        matricula = self.id_input.text()
        if not matricula:
            print("Digite a matrícula antes de capturar a foto.")
            return

        # Define o caminho para salvar a foto
        nome_arquivo = f"fotos/{matricula}.jpg"

        # Captura a foto com a câmera
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Erro ao acessar a câmera.")
                break
            cv2.imshow("Captura de Foto", frame)
            # Pressione 's' para salvar a foto
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite(nome_arquivo, frame)
                print(f"Foto salva com sucesso em {nome_arquivo}")
                break

        # Libera a câmera e fecha a janela
        camera.release()
        cv2.destroyAllWindows()


    def salvar_dados(self):
        nome = self.name_input.text()
        matricula = self.id_input.text()
        foto_path = f"fotos/{matricula}.jpg"

        # Verifica se os campos estão preenchidos
        if not nome or not matricula:
            print("Por favor, preencha todos os campos antes de salvar.")
            return

        try:
            # Conectando ao banco de dados
            conexao = sqlite3.connect("meu_banco.db")
            cursor = conexao.cursor()

            # Inserindo os dados no banco
            cursor.execute("""
            INSERT INTO usuarios (nome, matricula, foto_path) 
            VALUES (?, ?, ?)
            """, (nome, matricula, foto_path))

            # Salvando as alterações
            conexao.commit()
            print(f"Dados salvos com sucesso!\nNome: {nome}, Matrícula: {matricula}, Foto: {foto_path}")
        except sqlite3.IntegrityError:
            print("Erro: A matrícula já está cadastrada.")
        except Exception as e:
            print(f"Erro ao salvar os dados: {e}")
        finally:
            # Fechando a conexão com o banco de dados
            conexao.close()

        QMessageBox.information(self, "Sucesso!!", "O Aluno(a) foi salvo!")
        self.return_to_menu()
