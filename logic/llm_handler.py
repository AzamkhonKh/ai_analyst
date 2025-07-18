import traceback
import os
import json
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# LangChain Imports
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
# Import plotting interface and registry from plotting.py
from .plotting import PlotRegistry, ffthist_plot, histogram_plot, plot_all_ffthist, plot_all_histograms, plot_all_scatter_plots, plot_all_timeseries, plot_shap_feature_force, scatter_plot, timeseries_plot


class LLMHandler(QObject):

    def get_response_with_context(self, user_input: str, room_name: str, file_path: str = None, save_history: bool = True):
        """
        Handles user input with extra context (room, file). Decides whether to answer with a message or perform an action (e.g., plotting, modeling).
        """
        try:
            self.emit_status("Thinking (with context and file)...")
            # Get previous 3 messages for this room (if any)
            history = self.get_session_history(room_name)
            prev_msgs = []
            # prev_msgs = history.messages[-3:] if len(
            #     history.messages) >= 3 else history.messages[:]
            prev_msgs_text = "\n".join([
                f"{msg.type.capitalize()}: {msg.content}" for msg in prev_msgs
            ]) if prev_msgs else ""

            # Improved prompt for LLM intent and action detection
            prompt_value = (
                f"You are an expert assistant for a data analysis app.\n"
                f"User prompt: {user_input}\n"
            )
            if prev_msgs_text:
                prompt_value += f"Previous messages (most recent last):\n{prev_msgs_text}\n"
            if file_path:
                prompt_value += f"A file named '{os.path.basename(file_path)}' is available.\n"
            # Add possible features (columns) if available
            if room_name in self.dataframes:
                columns = list(self.dataframes[room_name].columns)
                features_hint = f"- Possible features (columns) in the dataset: {columns}\n"
            else:
                features_hint = ""
            prompt_value += (
                f"""
Your task is to determine the user's intent and respond ONLY with a JSON object in the following format:
{{
  "action": "plot" | "ml_model" | "chat",
  "plot_type": <type, if action is plot, e.g. 'histogram', 'scatter', 'timeseries', 'frequency_domain' or null>,
  "features": [<list of feature/column names, if relevant, else empty list>],
  "model_type": <type, if action is ml_model, e.g. 'logistic_regression', 'decision_tree', or null>,
  "explanation": <short explanation of your reasoning>
}}

Actions:
- If the user wants to visualize data (e.g., plot, diagram, chart), set action to "plot" and specify plot_type and features.
- If the user wants to create, train, or use a machine learning model, set action to "ml_model" and specify model_type and features.
- If the user just wants to chat or ask a question, set action to "chat" and leave other fields null or empty.
{features_hint}
Examples:
User: Show a histogram of column 'age'.
Response: {{"action": "plot", "plot_type": "histogram", "features": ["age"], "model_type": null, "explanation": "User requested a histogram plot of 'age'."}}

User: Train a decision tree to classify health status.
Response: {{"action": "ml_model", "plot_type": null, "features": ["health_status"], "model_type": "decision_tree", "explanation": "User wants to train a decision tree model for health status classification."}}

User: What is a neural network?
Response: {{"action": "chat", "plot_type": null, "features": [], "model_type": null, "explanation": "User is asking a general question."}}

Respond ONLY with the JSON object, no extra text.
"""
            )
            # Use LLM to analyze intent
            intent_response = self.llm.invoke(prompt_value)
            self.emit_status(intent_response)
            intent_text = intent_response.content if hasattr(
                intent_response, 'content') else str(intent_response)

            # Try to parse the LLM's response as JSON
            try:
                intent_json = json.loads(intent_text)
            except Exception as e:
                self.error_occurred.emit(
                    f"Could not parse LLM intent as JSON: {e}\nRaw response: {intent_text}")
                self.get_response(user_input, room_name)
                return

            action = intent_json.get("action")
            plot_type = intent_json.get("plot_type")
            features = intent_json.get("features") or []
            # If user asks for all features, treat as empty (meaning all)
            if isinstance(features, list) and len(features) == 1 and str(features[0]).strip().lower() in {"all", "*", "all features"}:
                features = []
            model_type = intent_json.get("model_type")
            explanation = intent_json.get("explanation")

            # Handle plotting
            match action:
                case "plot":
                    if not file_path or room_name not in self.dataframes:
                        self.response_ready.emit(
                            "No file loaded for plotting.")
                        return
                    df = self.dataframes[room_name]
                    if not plot_type:
                        self.response_ready.emit(
                            "Could not determine plot type or features. Please specify the column name and plot type.")
                        return
                    # Only use the first feature for now (extend as needed)
                    html = self.handle_plot(plot_type, df, features)
                    self.response_ready.emit(html)
                    if save_history:
                        history = self.get_session_history(room_name)
                        history.add_user_message(user_input)
                        history.add_ai_message(html)
                case "ml_model":
                    msg = f"[ML MODEL] Would create model '{model_type}' using features {features}. (Not yet implemented)"
                    self.response_ready.emit(msg)
                    if save_history:
                        history = self.get_session_history(room_name)
                        history.add_user_message(user_input)
                        history.add_ai_message(msg)
                case "chat":
                    self.get_response(user_input, room_name,
                                      save_history=save_history)
                    return
                case default:
                    self.response_ready.emit(
                        f"Unknown action: {action}. Explanation: {explanation}")
                    if save_history:
                        history = self.get_session_history(room_name)
                        history.add_user_message(user_input)
                        history.add_ai_message(
                            f"Unknown action: {action}. Explanation: {explanation}")
            return
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(
                f"LLM Action Error: {e}\nFile: {__file__}\nTraceback:\n{tb}")

    def _extract_feature(self, user_input: str, lower_intent: str, df: pd.DataFrame) -> str:
        """Extract feature/column name from user input or LLM intent."""
        for col in df.columns:
            if col.lower() in user_input.lower():
                return col
        for col in df.columns:
            if col.lower() in lower_intent:
                return col
        return None

    def handle_plot(self, plot_type: str, df: pd.DataFrame, features: str | list | None) -> str:
        """
        Extensible plot handler using registry. Add new plot types by registering them.
        Returns HTML string for the plot or error message.
        """
        try:
            plot_entry = self.plot_registry.get(plot_type)
            if plot_entry is None:
                return f"Plot type '{plot_type}' is not supported."

            # Determine which plot function to use
            if features is None or (isinstance(features, list) and len(features) == 0):
                plot_func = plot_entry.get('all')
            else:
                plot_func = plot_entry.get('feature')
            if plot_func is None:
                return f"Plot function for type '{plot_type}' is not registered."

            # Call the plot function efficiently
            if plot_type == "histogram":
                if isinstance(features, list) and features:
                    return "".join([plot_func(df=df, feature=f) for f in features])
                elif features:
                    return plot_func(df=df, feature=features)
                else:
                    return plot_func(df=df)
            elif plot_type == "scatter":
                if isinstance(features, list) and len(features) >= 2:
                    result = ""
                    for idx, f1 in enumerate(features):
                        for f2 in features[idx+1:]:
                            result += plot_func(df, f1, f2)
                    return result
                else:
                    return "I need at least 2 features to make a scatter diagram."
            elif plot_type == "timeseries" or plot_type == "frequency_domain":
                return plot_func(df=df, feature=features)
            else:
                return plot_func(df=df)
        except Exception as e:
            self.error_occurred.emit(f"Plotting Error: {e}")
            return f"Plotting Error: {e}"
    """
    Handles all LLM-related operations, including chat and document retrieval.
    Runs on a separate thread to avoid freezing the GUI.
    """
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    file_processed = pyqtSignal(str)
    csv_analyzed = pyqtSignal(str, bytes)  # analysis text, image bytes

    def __init__(self, model_name="llama3.1:8b"):
        super().__init__()
        self.chat_histories = {}  # Store message history for each room
        self.dataframes = {}     # Store loaded DataFrames for each room
        self.model_name = model_name
        self.plot_registry = PlotRegistry()
        self.plot_registry.register(
            "histogram", histogram_plot, plot_all_histograms)
        self.plot_registry.register(
            "scatter", scatter_plot, plot_all_scatter_plots)
        self.plot_registry.register(
            "timeseries", timeseries_plot, plot_all_timeseries)
        self.plot_registry.register(
            "frequency_domain", ffthist_plot, plot_all_ffthist)
        self.llm = None
        self.rag_chain_with_history = None
        self._init_llm_and_chain()

    def _init_llm_and_chain(self):
        """Initialize the LLM and the chat chain with message history."""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an expert assistant. Answer the user's questions. If you don't know, say so."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])

            chain = prompt | self.llm
            self.rag_chain_with_history = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
        except Exception as e:
            self.rag_chain_with_history = None
            self.error_occurred.emit(
                f"Initialization Error: {e}\n\nPlease ensure Ollama is running.")

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        Retrieves or creates a chat history for a given session (room).
        """
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]

    @pyqtSlot(str, str)
    def process_file(self, file_path: str, room_name: str):
        """
        Loads a text, CSV, or Excel file, stores DataFrame, and emits detected headers and shape.
        """
        try:
            file_name = os.path.basename(file_path)
            self.emit_status(f"Processing file: {file_name}...")

            ext = os.path.splitext(file_name)[1].lower()
            if ext in [".csv", ".txt"]:
                df = pd.read_csv(file_path)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path)
            else:
                self.error_occurred.emit(f"Unsupported file type: {ext}")
                return

            self.dataframes[room_name] = df
            headers = list(df.columns)
            shape = df.shape

            msg = (
                f"File '{file_name}' loaded successfully!\n"
                f"Detected columns: {headers}\n"
                f"Dataset shape: {shape[0]} rows × {shape[1]} columns."
            )
            msg += plot_shap_feature_force(df=df)

            self.file_processed.emit(msg)
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(
                f"File Processing Error: {e}\nFile: {__file__}\nTraceback:\n{tb}")

    @pyqtSlot(str, str)
    def get_response(self, user_input: str, room_name: str, save_history: bool = True):
        """
        Handles user input, retrieves relevant context using the retriever, and gets a response from the LLM chain. If no context is loaded, initializes the room with a dummy retriever. Stores both user prompts and LLM responses in the chat history for each room.
        """
        try:
            self.emit_status("Thinking (with context)...")
            if self.rag_chain_with_history is None:
                self.error_occurred.emit(
                    "LLM is not initialized. Please ensure Ollama is running and try again.")
                return
            # Always use chat history for Q&A
            response = self.rag_chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": room_name}}
            )

            answer = response.content if hasattr(
                response, 'content') else str(response)
            self.response_ready.emit(answer)
            if save_history:
                history = self.get_session_history(room_name)
                history.add_user_message(user_input)
                history.add_ai_message(answer)
        except Exception as e:

            tb = traceback.format_exc()
            self.error_occurred.emit(
                f"LLM Error: {e}\nFile: {__file__}\nTraceback:\n{tb}")

    def emit_status(self, message: str):
        """Helper to emit a status update."""
        print(message)  # Also print to console for debugging
