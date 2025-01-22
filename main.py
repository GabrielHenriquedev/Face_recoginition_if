import sys
from PyQt5.QtWidgets import QApplication

from views.Menu_view import MenuFaceRecognition

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        menu = MenuFaceRecognition()
        menu.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)
