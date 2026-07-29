# PDF RAG System

This project is a basic RAG (Retrieval-Augmented Generation) system for retrieving information from PDF files.

The PDF files are loaded and split into smaller chunks. These chunks are converted into embeddings using the `all-MiniLM-L6-v2` model and stored in ChromaDB. When a user gives a query, the query is also converted into an embedding and ChromaDB retrieves the most relevant content from the PDF.

## Technologies Used

* Python
* LangChain
* ChromaDB
* Sentence Transformers
* PyMuPDF

## How It Works

1. Load PDF files using PyMuPDFLoader
2. Split the PDF content into smaller chunks
3. Generate embeddings using Sentence Transformers
4. Store the embeddings and document content in ChromaDB
5. Convert the user query into an embedding
6. Search ChromaDB for similar content
7. Return the most relevant chunks along with the source file and page number

## Model Used

`all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings for the document chunks and user queries.

## Project Flow

```text
PDF Files
   ↓
Load Documents
   ↓
Split into Chunks
   ↓
Generate Embeddings
   ↓
Store in ChromaDB
   ↓
User Query
   ↓
Retrieve Relevant Content
```

## Installation

```bash
pip install chromadb numpy sentence-transformers langchain-community langchain-text-splitters pymupdf
```

## Run

Place the PDF files inside the `data/pdf` folder and run:

```bash
python main.py
```

Currently, this project handles PDF loading, embedding, vector storage and retrieval. The next step is to connect an LLM to the retrieved content to generate the final response.
