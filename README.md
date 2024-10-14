# MedRAG-textbooks

MedRAG-textbooks is a medical question-answering system that utilizes retrieval-augmented generation (RAG) with a large language model (OpenAI), leveraging textbook content as its corpus. The system retrieves relevant information using a MedCPT-based retriever and answers medical multiple-choice questions from the MMUL dataset.

## Features

- **LLM**: OpenAI models for answering medical questions.
- **Dataset**: MMUL (Medical Multiple Choice Understanding Learning) dataset.
- **Corpus**: Medical textbooks.
- **Retriever**: MedCPT model for retrieval-based augmented generation.

## Requirements

- Python 3.8 or higher
- Git
- Google Cloud Platform (GCP) for deployment (optional)

## Installation

Follow the steps below to set up the environment and run the system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/superbabii/MedRAG-textbooks.git
cd MedRAG-textbooks
```

### Step 2: Set Up a Virtual Environment

Ensure Python 3.8 and `venv` are installed.

```bash
sudo apt install python3 python3-pip -y
sudo apt install python3.8-venv
```

```bash
sudo apt install unzip
```

Create and activate a virtual environment:

```bash
python3 -m venv medrag_env
source medrag_env/bin/activate
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 4: Run the System

Once the dependencies are installed, you can run the main program:

```bash
python3 main.py
```

## Project Structure

- `main.py`: Main entry point for running the MedRAG-textbooks system, responsible for answering questions.
- `medrag.py`: Implements the MedRAG class for retrieval-augmented generation using OpenAI's models and MedCPT as the retriever.
- `utils.py`: Contains utility functions and classes for embedding, indexing, and retrieval using FAISS and MedCPT.

## Usage

1. Ensure the virtual environment is activated.
2. Run `python3 main.py` to start answering questions using the MMUL dataset.

## Deployment

The system is built to run on local machines or cloud platforms like Google Cloud Platform (GCP). Ensure GCP is configured for cloud deployment if needed.

## License

This project is licensed under the MIT License.
```
