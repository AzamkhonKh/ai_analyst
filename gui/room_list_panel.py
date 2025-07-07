from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QHBoxLayout, QPushButton, QInputDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal

class RoomListPanel(QWidget):
    room_changed = pyqtSignal(str)
    room_deleted = pyqtSignal(str)
    room_added = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setMaximumWidth(250)
        room_label = QLabel("Chat Rooms")
        room_label.setFont(room_label.font())
        self.room_list = QListWidget()
        self.room_list.itemClicked.connect(self.on_room_selected)
        room_button_layout = QHBoxLayout()
        self.add_room_btn = QPushButton("New Room")
        self.add_room_btn.clicked.connect(self.on_add_room)
        self.del_room_btn = QPushButton("Delete Room")
        self.del_room_btn.clicked.connect(self.on_delete_room)
        room_button_layout.addWidget(self.add_room_btn)
        room_button_layout.addWidget(self.del_room_btn)
        self.layout.addWidget(room_label)
        self.layout.addWidget(self.room_list)
        self.layout.addLayout(room_button_layout)

    def add_room(self, name: str):
        if not self.room_list.findItems(name, Qt.MatchExactly):
            self.room_list.addItem(name)
            self.room_added.emit(name)

    def on_add_room(self):
        text, ok = QInputDialog.getText(self, 'New Chat Room', 'Enter room name:')
        if ok and text:
            self.add_room(text)
            self.room_list.setCurrentRow(text)

    def on_delete_room(self):
        current_item = self.room_list.currentItem()
        if not current_item:
            return
        if self.room_list.count() <= 1:
            QMessageBox.critical(self, "Error", "Cannot delete the last room.")
            return
        room_name = current_item.text()
        reply = QMessageBox.question(self, 'Delete Room',
                                     f"Are you sure you want to delete the room '{room_name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            row = self.room_list.row(current_item)
            self.room_list.takeItem(row)
            self.room_deleted.emit(room_name)
            if self.room_list.count() > 0:
                self.room_list.setCurrentRow(0)
                self.on_room_selected(self.room_list.item(0))

    def on_room_selected(self, item):
        self.room_changed.emit(item.text())

    def setCurrentRow(self, row):
        self.room_list.setCurrentRow(row)

    def current_room(self):
        item = self.room_list.currentItem()
        return item.text() if item else None

    def count(self):
        return self.room_list.count()

    def findItems(self, name, flags):
        return self.room_list.findItems(name, flags)

    def item(self, row):
        return self.room_list.item(row)

    def setEnabled(self, enabled: bool):
        self.room_list.setEnabled(enabled)
        self.add_room_btn.setEnabled(enabled)
        self.del_room_btn.setEnabled(enabled) 