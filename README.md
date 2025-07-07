# PyQt5 LLM Chat & Data Analysis GUI

## Overview

This application is an interactive chat-based GUI for data analysis, powered by PyQt5 and a local LLM (Large Language Model) backend. It allows users to:

- Chat with an AI assistant
- Upload datasets (CSV or text files)
- Request statistical analysis and visualizations via natural language
- View results (including plots) directly in the chat interface
- Manage multiple chat rooms, each with its own context and file

## Features

### 🗂️ Multi-Room Chat

- Create, switch, and delete chat rooms
- Each room maintains its own chat history and file context

### 📁 File Upload & Contextual Analysis

- Upload CSV or text files to a chat room
- The uploaded file becomes the context for that room
- The AI can answer questions and perform analysis based on the loaded file

### 💬 Natural Language Data Analysis

- Ask questions about the uploaded file in plain English
- Example: `What is the mean of column A?`
- Example: `Show the time series for KO samples`

### 📊 Interactive Plotting

- Request plots by describing them in chat
- Example: `Plot histogram of column X`
- Example: `Plot the ratio of A to B`
- Plots are generated using matplotlib and shown directly in the chat

### 📈 Ratio & Statistical Analysis

- Request ratio analysis between columns
- Example: `Show the ratio of column1 to column2`
- Get summary statistics, head, and description of the dataset

### 🧠 LLM-Powered Q&A

- If a file is loaded, the LLM uses it as context (RAG)
- If no file is loaded, the LLM answers general questions

### 🖥️ Modern PyQt5 GUI

- Responsive, resizable interface
- Status bar for feedback
- Clean separation of UI and logic for maintainability

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies** (in your virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Ollama or your LLM backend is running locally**
4. **Run the application:**

   ```bash
   python main.py
   ```

## Usage

1. Start the app: `python main.py`
2. Create or select a chat room
3. Click `Load File` to upload a CSV or text file
4. Type your analysis request (e.g., `plot histogram of column X`, `show the ratio of A to B`)
5. View results and plots in the chat

## Example Requests

- `Plot histogram of column temperature`
- `Show the ratio of pressure to volume`
- `What are the main differences between OK and KO samples?`
- `Describe the dataset`

## Project Structure

```dir
main.py
logic/
  llm_handler.py
  ...
gui/
  chat_window.py
  room_list_panel.py
  chat_display_panel.py
  input_panel.py
  ...
dataset/
resources/
tests/
```

## Requirements

- Python 3.8+
- PyQt5
- pandas, matplotlib
- langchain, ollama, faiss, etc. (see requirements.txt)

## License

MIT
