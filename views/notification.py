# Adicione estas novas importações no início do arquivo
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QDesktopWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon


class NotificationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_position()

    def setup_ui(self):
        self.setWindowTitle("Alerta de Segurança")
        self.setFixedSize(300, 100)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel("Um desconhecido foi detectado!")
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                background-color: rgba(255, 68, 68, 0.9);
            }
        """)
        self.label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label)
        self.setLayout(layout)

    def setup_position(self):
        screen_geo = QDesktopWidget().availableGeometry()
        window_geo = self.geometry()
        x = screen_geo.width() - window_geo.width() - 20
        y = screen_geo.height() - window_geo.height() - 20
        self.move(x, y)

    def closeEvent(self, event):
        self.deleteLater()

