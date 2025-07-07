import os
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


class LLMHandler(QObject):
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
        self.retrievers = {}     # Store document retrievers for each room
        self.dataframes = {}     # Store loaded DataFrames for each room
        self.model_name = model_name
        try:
            # --- Core LLM and Embeddings Setup ---
            # Initialize the LLM
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0,
            )
            # Initialize the embeddings model (for turning text into vectors)
            self.embeddings = OllamaEmbeddings(model=self.model_name)

            # --- RAG (Retrieval-Augmented Generation) Chain Setup ---
            # This chain is for answering questions based on a loaded file.
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an expert assistant. Answer the user's questions based on the provided context. If the context doesn't have the answer, say so.\n\nContext:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(
                self.llm, rag_prompt)
            # We wrap this in a runnable with history to make it conversational
            self.rag_chain_with_history = RunnableWithMessageHistory(
                question_answer_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

        except Exception as e:
            # This will catch errors like Ollama not running
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
        Loads a text or CSV file, splits/analyzes it, and stores or emits results.
        """
        try:
            file_name = os.path.basename(file_path)
            self.emit_status(f"Processing file: {file_name}...")

            if file_path.lower().endswith('.csv'):
                self.process_csv_file(file_path, room_name)
                return

            # 1. Load the document
            loader = TextLoader(file_path)
            docs = loader.load()

            # 2. Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # 3. Create a vector store from the chunks
            vectorstore = FAISS.from_documents(
                documents=splits, embedding=self.embeddings)

            # 4. Create a retriever and store it for the current room
            self.retrievers[room_name] = vectorstore.as_retriever()
            self.emit_status(
                f"File '{file_name}' loaded and ready for questions in this room.")
            self.file_processed.emit(
                f"File '{file_name}' is now context for this room.")
        except Exception as e:
            self.error_occurred.emit(f"File Processing Error: {e}")

    def process_csv_file(self, file_path: str, room_name: str):
        """
        Loads a CSV file, analyzes it, and emits analysis and a diagram.
        """
        try:
            df = pd.read_csv(file_path)
            self.dataframes[room_name] = df  # Store for later analysis
            analysis = f"CSV Analysis for {os.path.basename(file_path)}\n"
            analysis += f"Shape: {df.shape}\n\n"
            analysis += f"Columns: {list(df.columns)}\n\n"
            analysis += f"Head:\n{df.head().to_string()}\n\n"
            analysis += f"Describe:\n{df.describe().to_string()}\n"
            # Plot: histogram of first numeric column
            numeric_cols = df.select_dtypes(include='number').columns
            img_bytes = b''
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                plt.figure(figsize=(6, 4))
                df[col].hist(bins=20)
                plt.title(f"Histogram of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img_bytes = buf.read()
                buf.close()
                self.emit_status(
                    f"CSV file loaded. Analysis and diagram ready.")
            else:
                self.emit_status(
                    f"CSV file loaded. No numeric columns for plotting.")
            self.csv_analyzed.emit(analysis, img_bytes)
            self.file_processed.emit(
                f"CSV file '{os.path.basename(file_path)}' analyzed and diagram generated.")
        except Exception as e:
            self.error_occurred.emit(f"CSV Processing Error: {e}")

    @pyqtSlot(str, str)
    def get_response(self, user_input: str, room_name: str):
        """
        Handles user input, retrieves relevant context using the retriever, and gets a response from the LLM chain. If no context is loaded, initializes the room with a dummy retriever. Stores both user prompts and LLM responses in the chat history for each room.
        """
        try:
            self.emit_status("Thinking (with context)...")
            if room_name not in self.retrievers:
                # Initialize with a dummy document to avoid errors
                from langchain_core.documents import Document
                dummy_doc = Document(page_content=" ", metadata={"room": room_name})
                vectorstore = FAISS.from_documents(
                    documents=[dummy_doc], embedding=self.embeddings)
                self.retrievers[room_name] = vectorstore.as_retriever()
            retriever = self.retrievers[room_name]
            # Use invoke instead of get_relevant_documents (deprecation fix)
            relevant_docs = retriever.invoke(user_input)
            # Run the chain with the retrieved context
            chain = self.rag_chain_with_history
            response = chain.invoke(
                {"input": user_input, "context": relevant_docs},
                config={"configurable": {"session_id": room_name}}
            )
            print("Chain response:", response)  # Debugging: see what keys are present
            # Fix: Only call .get() if response is a dict
            if isinstance(response, dict):
                answer = response.get('answer') or response.get('output') or str(response)
            else:
                answer = str(response)
            self.response_ready.emit(answer)
            # Store prompt and response in chat history
            history = self.get_session_history(room_name)
            history.add_user_message(user_input)
            history.add_ai_message(answer)
        except Exception as e:
            self.error_occurred.emit(f"LLM Error: {e}")

    def emit_status(self, message: str):
        """Helper to emit a status update."""
        print(message)  # Also print to console for debugging
