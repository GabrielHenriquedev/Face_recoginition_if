from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar


class LoadingScreen(QDialog):
    def __init__(self, total_steps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Carregando...")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)

        # Layout da tela de loading
        layout = QVBoxLayout()

        # Rótulo para exibir uma mensagem
        self.label = QLabel("Processando dados, aguarde...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Barra de progresso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.timer = QTimer()

    def update_progress(self, value):
        """Atualiza o valor da barra de progresso."""
        self.progress_bar.setValue(value)

    def showEvent(self, event):
        """Força a renderização da tela de loading antes do início do processamento."""
        super().showEvent(event)
        self.repaint()  # Força a renderização da tela de loading
