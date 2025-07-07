from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QFont

class ChatDisplayPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 15))
        layout.addWidget(self.chat_display)

    def append_message(self, sender: str, message: str):
        message_newline = message.replace("\n", '<br>')
        formatted_message = f"<b>{sender}:</b><br>{message_newline}<br><br>"
        self.chat_display.append(formatted_message)
        self.chat_display.ensureCursorVisible() # Auto-scroll

    def clear(self):
        self.chat_display.clear() 