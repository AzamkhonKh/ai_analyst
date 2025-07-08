from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QMessageBox
from PyQt5.QtCore import Qt, QThread
from logic.llm_handler import LLMHandler
from gui.room_list_panel import RoomListPanel
from gui.chat_display_panel import ChatDisplayPanel
from gui.input_panel import InputPanel
import base64

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 LLM Chat")
        self.setGeometry(100, 100, 1000, 700)
        self.current_room = "General"
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
        self.llm_handler.csv_analyzed.connect(self.display_csv_analysis)
        self.llm_thread.start()

    def init_ui(self):
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

    def on_send(self, user_text=None):
        if user_text is None:
            user_text = self.input_panel.user_input.text().strip()
        if not user_text:
            return
        self.chat_display_panel.append_message("You", user_text)
        self.toggle_inputs(False)
        self.update_status("LLM is thinking...")
        self.llm_handler.get_response(user_text, self.current_room)

    def on_load_file(self):
        from PyQt5.QtWidgets import QFileDialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text/CSV Files (*.txt *.csv);;All Files (*)", options=options)
        if file_path:
            self.toggle_inputs(False)
            self.update_status(f"Processing file: {file_path}...")
            self.llm_handler.process_file(file_path, self.current_room)

    def handle_response(self, response_text: str):
        self.chat_display_panel.append_message("LLM", response_text)
        self.toggle_inputs(True)
        self.update_status("Ready.")

    def display_csv_analysis(self, analysis_text: str, image_bytes: bytes):
        message_html = analysis_text
        if image_bytes:
            b64_data = base64.b64encode(image_bytes).decode('utf-8')
            img_tag = f'<br><img src=\"data:image/png;base64,{b64_data}\" style=\"max-width: 400px; max-height: 300px;\"/>'
            message_html += img_tag
        self.chat_display_panel.append_message("LLM", message_html)
        self.toggle_inputs(True)
        self.update_status("CSV analysis displayed.")

    def on_room_deleted(self, room_name: str):
        if room_name in self.llm_handler.chat_histories:
            del self.llm_handler.chat_histories[room_name]
        if room_name in self.llm_handler.retrievers:
            del self.llm_handler.retrievers[room_name]

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
        if self.current_room in self.llm_handler.retrievers:
            self.update_status(f"Ready. Context file is active in this room.")
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