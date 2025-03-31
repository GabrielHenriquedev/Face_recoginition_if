MAIN_STYLES = """
QWidget {
    background-color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #333333;
}

QLabel {
    font-size: 12pt;
    margin: 5px 0;
}

QLineEdit {
    font-size: 12pt;
    padding: 8px;
    border: 1px solid #cccccc;
    border-radius: 4px;
    margin: 5px 0;
}

QPushButton {
    font-size: 11pt;
    padding: 10px 20px;
    border-radius: 5px;
    margin: 5px;
    min-width: 120px;
}

QMessageBox {
    background-color: #ffffff;
}

QMessageBox QLabel {
    color: #333333;
    font-size: 12pt;
}
"""

BUTTON_STYLES = {
    "primary": """
        QPushButton {
            background-color: #8ec73d;
            color: #ffffff;
            border: 2px solid #7eb32d;
        }
        QPushButton:hover {
            background-color: #7eb32d;
        }
    """,
    "danger": """
        QPushButton {
            background-color: #ee1620;
            color: #ffffff;
            border: 2px solid #dd0510;
        }
        QPushButton:hover {
            background-color: #dd0510;
        }
    """,
    "secondary": """
        QPushButton {
            background-color: #ffffff;
            color: #8ec73d;
            border: 2px solid #8ec73d;
        }
        QPushButton:hover {
            background-color: #f0f8e8;
        }
    """
}

PREVIEW_STYLES = {
    "active": """
        QLabel#preview_label {
            border: 2px solid #8ec73d;
            background-color: #ffffff;
        }
    """,
    "inactive": """
        QLabel#preview_label {
            border: 2px dashed #cccccc;
            background-color: #f8f8f8;
        }
    """
}