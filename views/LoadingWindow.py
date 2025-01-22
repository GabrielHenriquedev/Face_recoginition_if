from PyQt5.QtWidgets import QApplication, QVBoxLayout, QProgressBar, QLabel, QWidget
from PyQt5.QtCore import Qt
import os
import cv2
import pickle
import face_recognition

class ProgressBarWindow(QWidget):
    def __init__(self, total_steps):
        super().__init__()
        self.setWindowTitle("Carregando Rostos Conhecidos")
        self.setGeometry(300, 300, 400, 150)

        layout = QVBoxLayout()

        self.label = QLabel("Carregando imagens, aguarde...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(total_steps)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def update_progress(self, step):
        self.progress_bar.setValue(step)
