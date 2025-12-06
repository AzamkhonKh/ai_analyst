#!/usr/bin/env python3
"""
Enhanced LLM Handler with BERT NLU Integration
================================================

This is a modified version of llm_handler.py that integrates the trained
BERT-based NLU for fast, reliable intent/slot extraction.

Key Changes:
-----------
1. Added NLU module for fast first-pass intent detection
2. Multi-turn dialogue support with clarification handling
3. LLM fallback only for chat or low-confidence cases
4. Reduced latency from ~2-5s (LLM) to ~50ms (BERT)

Integration Steps:
-----------------
1. Copy this file to ai_analyst/logic/llm_handler.py
2. Copy nlu_module.py to ai_analyst/logic/nlu_module.py
3. Copy trained model to ai_analyst/models/nlu/
4. Update requirements.txt with: torch, transformers

Author: AI Analyst Project
"""

import traceback
import os
import json
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# LangChain Imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
import pandas as pd

# Import NLU module
try:
    from .nlu_module import NLUModule, NLUConfig, NLUResult
    NLU_AVAILABLE = True
except ImportError:
    NLU_AVAILABLE = False
    print("Warning: NLU module not available. Using LLM-only mode.")

# Import plotting interface
from .plotting import (
    PlotRegistry, ffthist_plot, histogram_plot, plot_all_ffthist, 
    plot_all_histograms, plot_all_scatter_plots, plot_all_timeseries, 
    plot_shap_feature_force, scatter_plot, timeseries_plot
)


class LLMHandler(QObject):
    """
    Enhanced LLM Handler with integrated BERT NLU.
    
    Processing Pipeline:
    -------------------
    1. User input → BERT NLU (fast intent/slot extraction)
    2. If confidence > threshold AND complete → Execute action
    3. If incomplete → Ask clarification question
    4. If low confidence OR chat intent → Use LLM
    
    Benefits:
    - 10-100x faster response for plot/model requests
    - Works offline (no LLM needed for common requests)
    - Consistent, deterministic behavior
    - Multi-turn dialogue support
    """
    
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    file_processed = pyqtSignal(str)
    clarification_needed = pyqtSignal(str)  # New signal for clarification
    
    def __init__(self, model_name="llama3.1:8b"):
        super().__init__()
        self.chat_histories = {}
        self.dataframes = {}
        self.model_name = model_name
        self.plot_registry = PlotRegistry()
        self._register_plots()
        
        # Initialize NLU module
        self.nlu = None
        self.nlu_confidence_threshold = 0.7
        if NLU_AVAILABLE:
            self._init_nlu()
        
        # Initialize LLM
        self.llm = None
        self.rag_chain_with_history = None
        self._init_llm_and_chain()
    
    def _register_plots(self):
        """Register all plot types."""
        self.plot_registry.register("histogram", histogram_plot, plot_all_histograms)
        self.plot_registry.register("scatter", scatter_plot, plot_all_scatter_plots)
        self.plot_registry.register("timeseries", timeseries_plot, plot_all_timeseries)
        self.plot_registry.register("frequency_domain", ffthist_plot, plot_all_ffthist)
    
    def _init_nlu(self):
        """Initialize the BERT NLU module."""
        try:
            # Look for trained model in standard locations
            model_paths = [
                "models/nlu/model.pt",
                "../models/nlu/model.pt",
                os.path.expanduser("~/.ai_analyst/models/nlu/model.pt"),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            config = NLUConfig(
                model_path=model_path,
                confidence_threshold=self.nlu_confidence_threshold
            )
            self.nlu = NLUModule(config)
            self.emit_status("NLU module initialized successfully.")
            
        except Exception as e:
            self.emit_status(f"NLU initialization warning: {e}")
            self.nlu = None
    
    def _init_llm_and_chain(self):
        """Initialize the LLM and the chat chain with message history."""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an expert data analysis assistant. Answer the user's questions. "
                 "If you don't know, say so. Be concise and helpful."),
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
                f"LLM Initialization Error: {e}\n\nPlease ensure Ollama is running.")
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Retrieves or creates a chat history for a given session (room)."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]
    
    def get_response_with_context(
        self, 
        user_input: str, 
        room_name: str, 
        file_path: str = None, 
        save_history: bool = True
    ):
        """
        Enhanced response handler with NLU integration.
        
        Flow:
        1. Fast NLU pass for intent/slot extraction
        2. Route based on intent and confidence
        3. Handle clarifications for incomplete requests
        4. Fallback to LLM for chat or low confidence
        """
        try:
            self.emit_status("Processing...")
            
            # ================================================================
            # STEP 1: Fast NLU Pass
            # ================================================================
            if self.nlu is not None:
                # Check if we're in an ongoing multi-turn dialogue
                dialogue_state = self.nlu.get_dialogue_state(room_name)
                in_dialogue = (dialogue_state.intent is not None 
                              and not dialogue_state.is_complete)
                
                nlu_result = self.nlu.understand(user_input, room_name)
                self.emit_status(
                    f"NLU: {nlu_result.intent} (conf: {nlu_result.confidence:.2f})"
                )
                
                # High confidence action intent OR continuing a multi-turn dialogue
                if ((nlu_result.confidence >= self.nlu_confidence_threshold 
                    or in_dialogue)  # <-- Continue dialogue even with low confidence
                    and nlu_result.intent in ["plot", "ml_model"]):
                    
                    # Check if request is complete
                    if nlu_result.is_complete:
                        # Execute the action directly
                        self._execute_action(
                            nlu_result, room_name, file_path, 
                            user_input, save_history
                        )
                        return
                    else:
                        # Ask clarification question
                        if nlu_result.clarification_question:
                            self.response_ready.emit(nlu_result.clarification_question)
                            if save_history:
                                history = self.get_session_history(room_name)
                                history.add_user_message(user_input)
                                history.add_ai_message(nlu_result.clarification_question)
                            return
                
                # Chat intent (not in dialogue) or low confidence - use LLM
                if nlu_result.intent == "chat" and not in_dialogue:
                    self.emit_status("Thinking...")
                    self.get_response(user_input, room_name, save_history=save_history)
                    return
                
                # Low confidence and not in dialogue - use LLM  
                if nlu_result.confidence < self.nlu_confidence_threshold and not in_dialogue:
                    self.emit_status("Thinking...")
                    self.get_response(user_input, room_name, save_history=save_history)
                    return
            
            # ================================================================
            # STEP 2: LLM Fallback (if NLU not available)
            # ================================================================
            self._llm_intent_detection(user_input, room_name, file_path, save_history)
            
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(
                f"LLM Action Error: {e}\nFile: {__file__}\nTraceback:\n{tb}")
    
    def _execute_action(
        self, 
        nlu_result: 'NLUResult', 
        room_name: str, 
        file_path: str,
        user_input: str,
        save_history: bool
    ):
        """Execute an action based on NLU result."""
        
        action = nlu_result.intent
        slots = nlu_result.slots
        
        if action == "plot":
            if not file_path or room_name not in self.dataframes:
                self.response_ready.emit(
                    "No file loaded. Please load a data file first.")
                return
            
            df = self.dataframes[room_name]
            plot_type = slots.get("plot_type", "histogram")
            
            # Get features - prefer "features" list for scatter plots, fall back to "feature"
            features = slots.get("features", [])
            if not features and slots.get("feature"):
                features = [slots.get("feature")]
            
            html = self.handle_plot(plot_type, df, features)
            self.response_ready.emit(html)
            
            if save_history:
                history = self.get_session_history(room_name)
                history.add_user_message(user_input)
                history.add_ai_message(f"[Generated {plot_type} plot with features: {features}]")
            
            # Reset dialogue state after successful execution
            if self.nlu:
                self.nlu.reset_dialogue(room_name)
        
        elif action == "ml_model":
            model_type = slots.get("model_type", "logistic_regression")
            target = slots.get("target")
            features = slots.get("features", [])
            if not features and slots.get("feature"):
                features = [slots.get("feature")]
            
            msg = (
                f"<b>Training {model_type} model</b><br>"
                f"Target: {target}<br>"
                f"Features: {features if features else 'all'}<br>"
                f"<i>(Implementation pending)</i>"
            )
            self.response_ready.emit(msg)
            
            if save_history:
                history = self.get_session_history(room_name)
                history.add_user_message(user_input)
                history.add_ai_message(msg)
            
            if self.nlu:
                self.nlu.reset_dialogue(room_name)
    
    def _llm_intent_detection(
        self, 
        user_input: str, 
        room_name: str, 
        file_path: str, 
        save_history: bool
    ):
        """
        Original LLM-based intent detection (fallback).
        Used when NLU is not available or for complex queries.
        """
        history = self.get_session_history(room_name)
        prev_msgs = history.messages[-3:] if len(history.messages) >= 3 else history.messages[:]
        prev_msgs_text = "\n".join([
            f"{msg.type.capitalize()}: {msg.content}" for msg in prev_msgs
        ]) if prev_msgs else ""

        prompt_value = (
            f"You are an expert assistant for a data analysis app.\n"
            f"User prompt: {user_input}\n"
        )
        if prev_msgs_text:
            prompt_value += f"Previous messages (most recent last):\n{prev_msgs_text}\n"
        if file_path:
            prompt_value += f"A file named '{os.path.basename(file_path)}' is available.\n"
        
        if room_name in self.dataframes:
            columns = list(self.dataframes[room_name].columns)
            features_hint = f"- Possible features (columns) in the dataset: {columns}\n"
        else:
            features_hint = ""
        
        prompt_value += f"""
Your task is to determine the user's intent and respond ONLY with a JSON object:
{{
  "action": "plot" | "ml_model" | "chat",
  "plot_type": <type if action is plot, else null>,
  "features": [<list of feature names if relevant>],
  "model_type": <type if action is ml_model, else null>,
  "explanation": <short explanation>
}}
{features_hint}
Respond ONLY with the JSON object, no extra text.
"""
        
        intent_response = self.llm.invoke(prompt_value)
        intent_text = intent_response.content if hasattr(
            intent_response, 'content') else str(intent_response)

        try:
            intent_json = json.loads(intent_text)
        except Exception as e:
            self.get_response(user_input, room_name, save_history=save_history)
            return

        action = intent_json.get("action")
        plot_type = intent_json.get("plot_type")
        features = intent_json.get("features") or []
        model_type = intent_json.get("model_type")

        match action:
            case "plot":
                if not file_path or room_name not in self.dataframes:
                    self.response_ready.emit("No file loaded for plotting.")
                    return
                df = self.dataframes[room_name]
                if not plot_type:
                    self.response_ready.emit(
                        "Could not determine plot type. Please specify the column name and plot type.")
                    return
                html = self.handle_plot(plot_type, df, features)
                self.response_ready.emit(html)
                if save_history:
                    history.add_user_message(user_input)
                    history.add_ai_message(html)
            case "ml_model":
                msg = f"[ML MODEL] Would create model '{model_type}' using features {features}. (Not yet implemented)"
                self.response_ready.emit(msg)
                if save_history:
                    history.add_user_message(user_input)
                    history.add_ai_message(msg)
            case "chat":
                self.get_response(user_input, room_name, save_history=save_history)
            case _:
                self.response_ready.emit(f"Unknown action: {action}")
    
    def handle_plot(self, plot_type: str, df: pd.DataFrame, features: list) -> str:
        """Extensible plot handler using registry."""
        try:
            plot_entry = self.plot_registry.get(plot_type)
            if plot_entry is None:
                return f"Plot type '{plot_type}' is not supported."

            if features is None or (isinstance(features, list) and len(features) == 0):
                plot_func = plot_entry.get('all')
            else:
                plot_func = plot_entry.get('feature')
            
            if plot_func is None:
                return f"Plot function for type '{plot_type}' is not registered."

            if plot_type == "histogram":
                if isinstance(features, list) and features and features[0]:
                    return "".join([plot_func(df=df, feature=f) for f in features if f])
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
            elif plot_type in ["timeseries", "frequency_domain"]:
                if "time" not in df.columns:
                    return "Error: Dataset does not contain a 'time' column."
                return plot_func(df=df, feature=features, time_feature="time")
            else:
                return plot_func(df=df)
        except Exception as e:
            self.error_occurred.emit(f"Plotting Error: {e}")
            return f"Plotting Error: {e}"
    
    @pyqtSlot(str, str)
    def process_file(self, file_path: str, room_name: str):
        """Loads a data file and updates NLU with available features."""
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
            
            # Update NLU with available features
            if self.nlu:
                self.nlu.set_available_features(room_name, headers)

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
        """Get LLM response for chat intent."""
        try:
            self.emit_status("Thinking...")
            if self.rag_chain_with_history is None:
                self.error_occurred.emit(
                    "LLM is not initialized. Please ensure Ollama is running.")
                return
            
            response = self.rag_chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": room_name}}
            )

            answer = response.content if hasattr(response, 'content') else str(response)
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
        print(f"[LLMHandler] {message}")
