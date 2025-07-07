from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import pyqtSignal

class InputPanel(QWidget):
    send_clicked = pyqtSignal(str)
    load_file_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.returnPressed.connect(self.on_send)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        self.load_file_btn = QPushButton("Load File")
        self.load_file_btn.clicked.connect(self.on_load_file)
        layout.addWidget(self.user_input)
        layout.addWidget(self.send_btn)
        layout.addWidget(self.load_file_btn)

    def on_send(self):
        user_text = self.user_input.text().strip()
        if user_text:
            self.send_clicked.emit(user_text)
            self.user_input.clear()

    def on_load_file(self):
        self.load_file_clicked.emit()

    def setEnabled(self, enabled: bool):
        self.user_input.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.load_file_btn.setEnabled(enabled)

    def setPlaceholderText(self, text: str):
        self.user_input.setPlaceholderText(text) 