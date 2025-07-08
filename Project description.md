# Statistical AI Agent for Dataset Analysis

## Project Summary

This project focuses on creating an **AI Agent** capable of performing statistical analysis on datasets using machine learning and generative AI techniques. The primary goal is to develop a system that:

- Accepts datasets tagged as "OK" or "KO" (indicating different classes or states)
- Potentially trains a simple discriminating model (not necessarily pretrained)
- Automatically identifies and reports the most relevant statistical indices that differentiate between the two classes
- Interacts via a chat interface to generate plots (time series or frequency domain) of the dataset based on natural language requests
- Presents results (statistical indices, discrimination scores, and plots) through a GUI

## Problem Definition

Analyzing datasets to identify key distinguishing features between different classes (e.g., normal vs. anomalous, working vs. failed) often requires significant manual effort involving statistical computation and visualization. This project aims to automate parts of this process using an AI agent.

### Key Aspects and Challenges

- **Dataset Handling:** Ingesting and processing datasets with associated "OK"/"KO" labels.
- **Statistical Feature Identification:** Automatically calculating various statistical measures (mean, median, mode, standard deviation, variance, etc.) and determining which ones best separate the OK and KO groups. This might involve techniques like feature importance analysis or simple discriminative model training.
- **Model Training (Optional):** Exploring the use of a simple, trainable model (e.g., Logistic Regression, SVM, Decision Tree) to help identify discriminative features or provide a classification score.
- **AI Agent Interaction:** Designing an agent (e.g., using Python functions potentially orchestrated by an LLM like LLAMA-2/3) that can understand analysis requests.
- **Natural Language Plotting:** Interpreting chat-based requests (e.g., "plot the frequency spectrum of dataset X", "show the time series for KO samples") and generating the corresponding visualizations.
- **GUI Presentation:** Developing an interface to display the input options, statistical results, scores, and generated plots.
- **Local Deployment:** Setting up the necessary platform, including local AI models.

## Required Background

- Python programming skills and understanding of fundamental statistics and data analysis techniques
- Experience with machine learning concepts and libraries (e.g., Scikit-learn, Pandas, NumPy)
- Familiarity with data visualization libraries (e.g., Matplotlib, Seaborn)
- Basic understanding of AI/LLMs for chat interaction (optional but implied by "AI Agent" and chat requests)

## Working Environment

- Students can work on laptops or desktops (Unix-like or Windows)
- **Core Implementation:** Python, utilizing data analysis (Pandas, NumPy), machine learning (Scikit-learn), and plotting (Matplotlib/Seaborn) libraries
- **AI Backend (Optional):** Local installation of a generative AI package (e.g., LLAMA-2, LLAMA-3) if used for chat interpretation or agent orchestration
- **GUI:** Python GUI framework (e.g., Tkinter, PyQt, Streamlit, Dash)

## Deliverables

All projects must be delivered including the following material:

- **Source Code:** Well-commented Python code for the AI agent, data processing, statistical analysis, model training (if applicable), plotting functions, and GUI.
- **README File:** Instructions on required libraries, environment setup (including local AI model if used), how to run the application, and format of input datasets.
- **Documentation File:** A report (Word, LaTeX, Markdown, etc.) detailing:
  - **System Architecture:** Overall design of the agent and GUI.
  - **Design Choices:** Statistical measures implemented, feature selection/discrimination methods used, model choice (if any), approach to interpreting chat requests, GUI design decisions.
  - **Experimental Evaluation:** Demonstrating the system on sample datasets (OK/KO). Evaluation of the effectiveness of the identified statistical indices in discriminating classes. Examples of chat requests and generated plots. Discussion of accuracy and limitations.
- **Presentation:** Slides (PowerPoint or similar) for a final project presentation (approx. 15 minutes) outlining the project's motivation, design, functionality, results, and potential improvements.

## Use Cases

### 1. User chatts with system

<!--
actor User
participant "GUI Chat Room" as GUI
participant "Backend" as BE
participant "Local LLM" as LLM

User -> GUI: Sends a message
GUI -> BE: 
BE -> LLM: Provide context, history of chat \n (Set ups tone of AI agent)
LLM -> BE: 
BE -> GUI: Format response in html
GUI -> User: Display response \n from AI agent
-->

!["UC chatting"](/docs/UCChat.png)

### 2. User Uploads Dataset, Views Statistics

<!-- PlantUML
actor User
participant "GUI Chat Room" as GUI
participant "Backend" as BE
participant "Local LLM" as LLM

User -> GUI: Upload dataset file (csv, xlsx)
GUI -> BE: Send dataset file
BE -> BE: Calculate statistics (median, max, min, std)
BE -> GUI: Send statistics message
GUI -> User: Display statistics in chat
-->
!["UC upload dataset"](/docs/UCUploadFile.png)

### 3. User asks for plotting diagram

<!--
actor User
participant "GUI Chat Room (File uploaded)" as GUI
participant "Backend" as BE
participant "Local LLM" as LLM

User -> GUI: Request possible plots (e.g. histogram) for feature X
GUI -> BE: Send request ("Draw histogram for feature X")
BE -> LLM: Extract keywords ("draw", "histogram", feature X)
BE -> BE: Generate histogram plot
BE -> GUI: Send plot image
GUI -> User: Display plot in chat
-->
!["UC plot diagram"](/docs/UCPlotDiagram.png)
