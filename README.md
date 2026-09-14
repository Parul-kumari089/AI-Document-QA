# AI Document Q&A using RAG

An AI-powered document question-answering system that allows users to upload a PDF and ask questions about its content.

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the document and generate answers using a locally running Llama 3.2 model through Ollama.

## Features

- Upload a PDF document
- Extract text from PDF pages
- Split text into smaller chunks
- Generate embeddings using Sentence Transformers
- Store embeddings using FAISS
- Retrieve relevant document chunks for a question
- Generate answers using Llama 3.2 locally through Ollama
- Display the source pages used for the answer
- Streamlit-based user interface

## Tech Stack

- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Ollama
- Llama 3.2 3B

## Project Structure

```text
AI-Document-QA/
│
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
