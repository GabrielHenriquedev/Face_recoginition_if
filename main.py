import sys
from PyQt5.QtWidgets import QApplication

from views.Menu_view import MenuFaceRecognition

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MenuFaceRecognition()
    menu.show()  # Exibe o menu antes de qualquer outra execução
    sys.exit(app.exec_())
