# AI Document Q&A using RAG

AI Document Q&A is a Retrieval-Augmented Generation (RAG) based application that allows users to upload a PDF document and ask questions about its content.

The system retrieves the most relevant information from the document and uses a locally running Llama 3.2 model to generate an answer.

## Features

- Upload PDF documents
- Extract text from PDF
- Split document text into smaller chunks
- Generate text embeddings
- Store embeddings using FAISS
- Retrieve relevant document chunks
- Generate answers using Llama 3.2
- Display source pages used for the answer
- Simple Streamlit interface

## Technologies Used

- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Ollama
- Llama 3.2 3B

## How the Project Works

The project follows a RAG pipeline:

PDF Document
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
User Question
↓
Question Embedding
↓
Relevant Chunks Retrieval
↓
Llama 3.2
↓
Answer + Sources

## Project Structure

AI-Document-QA/

├── data/
│   └── documents/
│       └── employee_handbook.pdf
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── qa.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── vectorstore/
│   ├── chunks.json
│   └── document.index
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

## Installation

Clone the repository:

git clone https://github.com/Parul-kumari089/AI-Document-QA.git

Go to the project directory:

cd AI-Document-QA

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

## Ollama Setup

This project uses Ollama to run the Llama 3.2 3B model locally.

Pull the model using:

ollama pull llama3.2:3b

Make sure Ollama is running before starting the application.

## Run the Application

Run the Streamlit application:

streamlit run app.py

The application will open in your browser.

Upload a PDF, process the document, and ask questions about its content.

## Example

Question:

What are the normal working hours?

Answer:

Normal working hours are from 9:00 AM to 6:00 PM, Monday through Friday.

The application also displays the source pages from which the relevant information was retrieved.

## RAG Implementation

The application does not directly provide the complete document to the language model.

Instead:

1. The PDF text is extracted.
2. The text is divided into smaller chunks.
3. Each chunk is converted into an embedding using Sentence Transformers.
4. The embeddings are stored in a FAISS vector store.
5. When a user asks a question, the question is converted into an embedding.
6. FAISS retrieves the most relevant chunks.
7. The retrieved chunks are provided as context to Llama 3.2.
8. The model generates an answer using the provided context.

The model is instructed to answer only from the retrieved document context.

## Author

Parul Luharuka
