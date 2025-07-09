from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QMessageBox
from PyQt5.QtCore import Qt, QThread
from logic.llm_handler import LLMHandler
from gui.room_list_panel import RoomListPanel
from gui.chat_display_panel import ChatDisplayPanel
from gui.input_panel import InputPanel
import base64
from enum import Enum

class Models(Enum):
    DEEPSEEK = "deepseek-r1:7b"
    LLAMA = "llama3.1:8b"


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 LLM Chat")
        self.setGeometry(100, 100, 1000, 700)
        self.current_room = "General"
        self.current_model = Models.LLAMA
        self.setup_backend()
        self.init_ui()
        self.room_panel.add_room(self.current_room)
        self.room_panel.setCurrentRow(0)
        self.update_status("Ready. Select a room and start chatting.")

    def setup_backend(self):
        self.llm_thread = QThread()
        self.llm_handler = LLMHandler()
        self.llm_handler.moveToThread(self.llm_thread)
        self.llm_handler.response_ready.connect(self.handle_response)
        self.llm_handler.error_occurred.connect(self.show_error)
        self.llm_thread.start()

    def init_ui(self):
        from PyQt5.QtWidgets import QComboBox, QLabel, QHBoxLayout as QHBox
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.room_panel = RoomListPanel()
        self.room_panel.room_changed.connect(self.switch_room)
        self.room_panel.room_deleted.connect(self.on_room_deleted)
        self.room_panel.room_added.connect(self.on_room_added)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.chat_display_panel = ChatDisplayPanel()
        right_layout.addWidget(self.chat_display_panel)
        # Model selector UI
        model_selector_layout = QHBox()
        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.addItem("Llama 3", Models.LLAMA.value)
        self.model_selector.addItem("DeepSeek", Models.DEEPSEEK.value)
        self.model_selector.setCurrentIndex(0)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        model_selector_layout.addWidget(model_label)
        model_selector_layout.addWidget(self.model_selector)
        right_layout.addLayout(model_selector_layout)
        self.input_panel = InputPanel()
        self.input_panel.send_clicked.connect(self.on_send)
        self.input_panel.load_file_clicked.connect(self.on_load_file)
        right_layout.addWidget(self.input_panel)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.room_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 800])
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Welcome to the LLM Chat!")

    def on_model_changed(self, idx):
        model_value = self.model_selector.currentData()
        self.current_model = Models(model_value)
        # Immediately update LLM handler model if possible
        if hasattr(self.llm_handler, 'llm') and hasattr(self.llm_handler.llm, 'model'):
            self.llm_handler.llm.model = model_value
        if hasattr(self.llm_handler, 'model_name'):
            self.llm_handler.model_name = model_value
        self.update_status(f"Model switched to: {self.current_model.name}")

    def on_send(self, user_text=None):
        if user_text is None:
            user_text = self.input_panel.user_input.text().strip()
        if not user_text:
            return
        self.chat_display_panel.append_message("You", user_text)
        self.toggle_inputs(False)
        self.update_status("LLM is thinking...")
        # Pass extra context: file path and room name
        file_path = getattr(self, 'room_files', {}).get(self.current_room, None)
        # If LLM handler supports extra context, pass it; else fallback
        if hasattr(self.llm_handler, 'get_response_with_context'):
            self.llm_handler.get_response_with_context(user_text, self.current_room, file_path)
        else:
            self.llm_handler.get_response(user_text, self.current_room)

    def on_load_file(self):
        from PyQt5.QtWidgets import QFileDialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Text/CSV Files (*.txt *.csv);;All Files (*)", options=options)
        if file_path:
            self.update_status(f"Processing file: {file_path}...")
            # Share file path and current room with LLM handler
            self.llm_handler.process_file(file_path, self.current_room)
            # Optionally, store last uploaded file for this room
            if not hasattr(self, 'room_files'):
                self.room_files = {}
            self.room_files[self.current_room] = file_path

    def handle_response(self, response_text: str):
        self.chat_display_panel.append_message("LLM", response_text)
        self.toggle_inputs(True)
        self.update_status("Ready.")

    def on_room_deleted(self, room_name: str):
        if room_name in self.llm_handler.chat_histories:
            del self.llm_handler.chat_histories[room_name]

    def on_room_added(self, room_name: str):
        pass

    def switch_room(self, room_name: str):
        self.current_room = room_name
        self.chat_display_panel.clear()
        self.setWindowTitle(f"PyQt5 LLM Chat - {self.current_room}")
        history = self.llm_handler.get_session_history(self.current_room)
        for msg in history.messages:
            sender = "You" if msg.type == "human" else "LLM"
            self.chat_display_panel.append_message(sender, msg.content)
        else:
            self.update_status("Ready.")

    def toggle_inputs(self, enabled: bool):
        self.input_panel.setEnabled(enabled)
        self.room_panel.setEnabled(enabled)

    def update_status(self, message: str):
        self.statusBar().showMessage(message)

    def show_error(self, error_message: str):
        QMessageBox.critical(self, "Error", error_message)
        self.toggle_inputs(True)
        self.update_status("Error occurred. Ready for new input.")

    def closeEvent(self, event):
        self.llm_thread.quit()
        self.llm_thread.wait()
        event.accept()
